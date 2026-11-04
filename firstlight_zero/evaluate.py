"""Zero snapshots vs a frozen V4 opponent checkpoint (`--opponent PATH`), and round robins between snapshots, on the X-Bow mirror.
Field names keep `il_` for the opponent side whichever opponent is used.

Protocol, recorded in every result:
- Opponent: argmax; zero policy: --small-mode (sample = T 1, argmax) or, in sweeps, a temperature per row (0 = argmax,
  the same temperature divides the gate, option and location logits).
- Both sides use the earliest legal delay with a 200 ms ceiling (bins 0/50/100/150/200). The opponent's delay mask is
  set explicitly after loading and checked.
- Forced opening for every owner: 15 s without a successful card -> one forced ACT, the policy picks card and cell.
- No execution jitter.
- `once`/`loop`: 20 games by default = 10 fixed seeds x both seats. `sweep`: many snapshots (and temperatures) in one
  plan, every (snapshot, temperature) plays the same seed list; `--schedule single` = one game per seed, seats alternate.

Engines listen on the loopback interface, so every engine gets one worker process on its own host (firstlight_zero/hosts.py).
Each engine hosts 8 resident envs; a worker plays up to `slots_per_engine` games at once on one batch coordinator.

Game records (games/w*.jsonl, one line per game) carry, besides result/crowns/tower hp:
- plays: per owner [tick, card_id, [x, y] in that owner's own frame, 0..1]
- events: both owners' executed plays [tick, owner, card_id, x, y, cost, own_elixir_before, enemy_elixir_before];
  x, y in native arena units (18000 x 32000, owner 0's frame; owner 1's own frame is (18000-x, 32000-y))
- ability: per owner Hero Knight ability activation ticks (public tracker)
- trace: every 20 ticks [tick, elixir0, elixir1, tower_hp0, tower_hp1, elixir_multiplier]
- board: every 20 ticks [tick, owner, kind, x, y, hp_ratio] for X-Bow (X), Tesla (T), Evo Tesla (TE), Hero Knight (K)
- elixir: per owner mean held elixir and share of decision frames at 10 (before the gameplay end tick)

    python firstlight_zero/evaluate.py once  --run RUN --opponent OPP.pt [--version 400]
    python firstlight_zero/evaluate.py loop  --run RUN --opponent OPP.pt --every 25 --games 10     # until RUN/STOP
    python firstlight_zero/evaluate.py sweep --run RUN --name NAME --opponent OPP.pt --versions 10:1500:10 --games 40 [--temps 1]
    python firstlight_zero/evaluate.py summarize-sweep --run RUN --name NAME
    python firstlight_zero/evaluate.py roundrobin --run RUN --name NAME --top-from SWEEP --top 10 --temps 0,0.5,0.7,1,1.3
"""
from pathlib import Path
import argparse, bisect, collections, hashlib, itertools, json, math, multiprocessing as mp, os, random, shutil, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from firstlight_zero.hosts import load_engines, spawn_on   # noqa: E402

DELAYS_MS = [0, 50, 100, 150, 200]
FORCE_AFTER_SECONDS = 15
SEED = 20261101
SWEEP_SEED = 20261103
SLOTS_PER_ENGINE = 8
TRACE_TICKS = 20
BOARD_KINDS = {'form:Xbow': 'X', 'form:Tesla': 'T', 'form:Tesla_EV1': 'TE', 'form:KnightHero': 'K'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = Path(f'{path}.{os.getpid()}.tmp'); tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)   # per-process tmp: workers race


def schedule(games, kind='paired', seed=SEED):
    """'paired': games//2 seeds x both seats (the loop's schedule). 'single': one game per seed, small seat alternates."""
    rng = random.Random(seed)
    if kind == 'paired':
        seeds = [rng.randrange(1, 2**31 - 1) for _ in range(games // 2)]
        return [dict(game=2 * i + k, seed=s, small_owner=k) for i, s in enumerate(seeds) for k in (0, 1)]
    seeds = [rng.randrange(1, 2**31 - 1) for _ in range(games)]
    return [dict(game=i, seed=s, small_owner=i % 2) for i, s in enumerate(seeds)]


def worker(index, engines, out, small_path, small_mode, device):
    """Plays every game queued on `engines` ([(spec, [game rows])], local engines) and appends one record per game."""
    import numpy as np
    import torch
    from native_runner.cr_native_env import ResidentNativeClashEnv
    from native_runner.resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1
    from native_runner.training.v4.ppo_runtime import ppo_environment_config_v4, build_ppo_environment_v4, episode_for_matchup_v4
    from native_runner.training.v4.deck_pool import DeckPoolManifestV4
    from native_runner.training.v4.matchmaking import ScheduledMatchupV4, POLICY_CURRENT
    from native_runner.training.v4.factory import build_episode_tensorizers_v4
    from native_runner.training.v4.decoding import decode_action_sequence_v4
    from native_runner.training.v4.tensors import GATE_ACT
    from native_runner.training.v4.policy_loader import _DeterministicCudaGraphV4
    from firstlight_zero.hosts import pin
    from firstlight_zero.opponent import load_opponent as load_il
    from firstlight_zero.reward import deadline_cancellation
    from firstlight_zero.spec import XBOW_DECK_ID, POOL, BATTLE_LEVEL, GAMEPLAY_END_TICK
    from firstlight_zero.features import Featurizer2
    from firstlight_zero.batch import stack_inputs
    from firstlight_zero.spec import KNIGHT, UNIT_KEYS
    from firstlight_zero.actions import native_action, wait_action
    from firstlight_zero.model import XbowNet
    from native_runner.contracts import ActionKind

    meta = json.loads((out / 'config.json').read_text())
    torch.set_num_threads(1); torch.manual_seed(SEED + index)
    pin(spec.get('cpu') for spec, _ in engines)
    device = torch.device(device)
    arch = json.loads((out / 'small-model.json').read_text()); models = {}

    def small_model(path):
        if path not in models:
            m = XbowNet(**arch).eval(); m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)); models[path] = m
        return models[path]
    default_temp = 0. if small_mode == 'argmax' else 1.
    remote = il_remote = None
    if meta.get('inference') == 'server' or meta.get('il_inference') == 'server':
        import socket
        from firstlight_zero.batch import pack_request
        from firstlight_zero.server import send_msg, recv_msg
        plan = json.loads((out / 'plan.json').read_text()); pos = [e['engine'] for e, _ in plan].index(index)

        def connect(registry):
            # consecutive plan positions share a server, so each zero server holds only a few snapshots at a time
            regs = sorted(Path(registry).glob('*.json')); reg = json.loads(regs[pos * len(regs) // len(plan)].read_text())
            s_ = socket.create_connection((reg['host'], reg['port']), timeout=600.); s_.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s_, reg

        def receive(sock_):
            reply = recv_msg(sock_)
            if 'error' in reply:
                raise RuntimeError('inference server: ' + reply['error'])
            return reply
    if meta.get('inference') == 'server':
        # zero seats go to the run's GPU inference servers (as in training). Model key = snapshot version.
        zero_sock, _ = connect(meta['server_registry'])
        remote = dict(send=lambda seats, temps: send_msg(zero_sock, pack_request(index, seats, temps=temps)),
                      recv=lambda n: [{k: r[k][j].item() for k in ('gate', 'option', 'cell')} for r in (receive(zero_sock),) for j in range(n)])
    il = None
    if meta.get('il_inference') == 'server' and any('opp_model' not in row for _, rows in engines for row in rows):
        # the IL-family opponent runs on the GPU IL servers (firstlight_zero/il_server.py); tensorize/record/decode stay here
        il_sock, il_reg = connect(meta['il_server_registry'])
        assert il_reg['protocol']['delay_policy'] == 'earliest' and il_reg['protocol']['allowed_delay_offsets_ms'] == DELAYS_MS, il_reg
        assert Path(il_reg['protocol']['checkpoint']) == Path(meta['il_checkpoint']), (il_reg, meta['il_checkpoint'])
        if not (out / 'il-protocol.json').exists():
            atomic(out / 'il-protocol.json', il_reg['protocol'])
        from firstlight_zero.il_server import to_wire, from_wire
        from native_runner.training.v4.tensors import concatenate_padded_tensor_records
        il_remote = dict(send=lambda keys, starts, forces, batches: send_msg(il_sock, dict(
                             client=index, keys=keys, starts=starts, forces=forces, batch=to_wire(concatenate_padded_tensor_records(tuple(batches))))),
                         recv=lambda: (lambda r: dict(r, actions=from_wire(r['actions'])))(receive(il_sock)))
    elif any('opp_model' not in row for _, rows in engines for row in rows):   # an IL-family opponent is needed, on this CPU
        il = load_il(Path(meta['il_checkpoint']), device, False)
        il.set_inference_delay_offset_max_ms(200)
        allowed = [int(v) for v, ok in zip(il.config.delay_offset_ms, il._inference_delay_offset_mask.cpu().tolist()) if ok]
        assert il.delay_policy == 'earliest' and not il.ppo_rollout_sampling and allowed == DELAYS_MS, (il.delay_policy, allowed)
        if index == engines[0][0]['engine'] and not (out / 'il-protocol.json').exists():
            atomic(out / 'il-protocol.json', dict(mode='argmax', delay_policy=il.delay_policy, allowed_delay_offsets_ms=allowed,
                                                  actual_delay_ceiling_ms=max(allowed), runtime_cap_ms=il._inference_delay_offset_max_ms))
    graph = _DeterministicCudaGraphV4(device) if device.type == 'cuda' else None
    deck = next(d for d in DeckPoolManifestV4.load(POOL).decks if d.deck_id == XBOW_DECK_ID)
    cfg = ppo_environment_config_v4(gamma_per_decision=.9997, shaping_beta=0.)
    reset_options = dict(render_mode='headless', decision_ticks=5, event_driven_decisions=False, reuse_runner_attestation=True)
    log = out / 'games' / f'w{index:03d}.jsonl'
    slots_cap = int(meta.get('slots_per_engine', 1))
    timer = collections.Counter()

    groups = []
    for spec, rows in engines:
        n = max(1, min(len(spec['env_ids']), slots_cap, len(rows)))
        natives = [ResidentNativeClashEnv(eid, '127.0.0.1', spec['port'], timeout=120.) for eid in spec['env_ids'][:n]]
        coord = ResidentBatchCoordinatorV1(tuple(natives), timeout=120., training_capture=True, compressed_json=False, validate_json_capture=False)
        lanes = []
        for nat in natives:
            env = build_ppo_environment_v4(ResidentBatchNativeProxyV1(nat, coord), cfg)
            env.simulated_card_delay_jitter_ticks = 0; env.prepare_runner_attestation()
            lanes.append(dict(native=nat, env_id=int(nat.env_id), env=env, game=None, slot=sum(len(gr['lanes']) for gr in groups) + len(lanes)))
        groups.append(dict(spec=spec, coord=coord, lanes=lanes, queue=list(rows)))
    pool = ThreadPoolExecutor(sum(len(gr['lanes']) for gr in groups))

    def start(gr, lane):
        row = gr['queue'].pop(0); s_owner = row['small_owner']; il_owner = 1 - s_owner
        m = ScheduledMatchupV4(sequence=row['game'] + 1, category='xbow-mirror-eval', policy_matchup=POLICY_CURRENT, deck0=deck, deck1=deck,
                               current_owners=(0, 1), battle_level=BATTLE_LEVEL, seed=row['seed'])
        layouts = m.runtime_layouts(); episode = episode_for_matchup_v4(m, cfg)
        obs, _ = lane['env'].reset(episode, options=reset_options)
        model = row.get('model', small_path); temp = float(row.get('temperature', default_temp))
        # seat -> zero policy (checkpoint path, temperature); the opponent seat is the IL-family model unless the row names an zero one
        pol = {s_owner: (model, temp, row.get('version'))}
        if 'opp_model' in row:
            pol[il_owner] = (row['opp_model'], float(row.get('opp_temperature', 1.)), row.get('opp_version'))
            tz = None
        else:
            tz = build_episode_tensorizers_v4(episode)[il_owner]
            elixir = {o: float(next(q for q in obs[o].players if q.owner == o).elixir_exact) for o in (0, 1)}
            tz.start_episode(obs[il_owner], initial_elixir=elixir if tz.tracker is not None else None)
        lane['game'] = dict(row=row, obs=obs, small=s_owner, il=il_owner, tz=tz, first=True, model=model, temp=temp, pol=pol,
                            fz={o: Featurizer2(o, episode, obs[o], layouts[o].card_ids) for o in (0, 1)},
                            state={o: small_model(p[0]).initial_state(1) for o, p in pol.items()} if remote is None else None,
                            fresh={o: True for o in pol},
                            il_state=il.initial_state(1, device=device) if tz is not None and il is not None else None,
                            ids=[set(), set()], rejected=[0, 0], cancelled=[0, 0], acts=[0, 0], first_act=[None, None],
                            forced=[None, None], started=time.time(),
                            elix=([], []), seen=set(), events=[], ability=([], []), trace=[], board=[], last_trace=-TRACE_TICKS,
                            # exact replay: the episode (seed, decks, rules) plus every submitted non-wait action at its decision tick
                            native=dict(matchup=dict(sequence=row['game'] + 1, category='xbow-mirror-eval', seed=row['seed'],
                                                     battle_level=BATTLE_LEVEL, deck_id=XBOW_DECK_ID),
                                        episode=episode.to_dict(), reset_options=reset_options,
                                        layouts=[dict(card_ids=list(layouts[o].card_ids), form_availability=list(layouts[o].form_availability),
                                                      tower_troop_id=layouts[o].tower_troop_id) for o in (0, 1)]),
                            actions=[], outcome={}, leak=[0., 0.], ability_events=[])

    def observe(g, obs, tick):
        """Per decision frame: exact elixir of both owners, newly executed plays (absolute position), Hero Knight activations,
        and every TRACE_TICKS a trace / board sample."""
        el = [float(next(q for q in obs[o].players if q.owner == o).elixir_exact) for o in (0, 1)]
        mult = float(obs[0].time.elixir_multiplier or 1.)
        for o in (0, 1):
            if g['elix'][o] and g['elix'][o][-1][1] >= 9.999 and el[o] >= 9.999 and tick < GAMEPLAY_END_TICK:
                g['leak'][o] += (tick - g['elix'][o][-1][0]) * mult / 56.   # 1 elixir per 2.8 s = 56 ticks at x1
            g['elix'][o].append((tick, el[o]))
        for o in (0, 1):
            for e in obs[o].events:
                if e.owner is None:
                    continue
                kind = (e.data or {}).get('kind')
                if e.event_type in ('ability_activation', 'runtime_ability_activation') or (e.event_type == 'action_executed' and kind == 'activate_ability'):
                    t_ = int(e.tick); own = int(e.owner)
                    if all(abs(t_ - x) > 2 for x in g['ability'][own]):    # one activation can surface as several event types
                        g['ability'][own].append(t_)
                        g['ability_events'].append([t_, own, e.event_type, kind, None if e.entity_id is None else int(e.entity_id)])
                    continue
                if e.event_type != 'action_executed' or e.card_id is None:
                    continue
                pos = tuple(int(round(float(v))) for v in (e.position or ()))
                key = (int(e.tick), int(e.owner), int(e.card_id), pos)
                if key in g['seen']:
                    continue
                g['seen'].add(key)
                cost = (e.data or {}).get('effective_cost')
                g['events'].append([int(e.tick), int(e.owner), int(e.card_id), pos[0] if pos else None, pos[1] if len(pos) > 1 else None,
                                    float(cost) if isinstance(cost, (int, float)) else None])
        if tick - g['last_trace'] >= TRACE_TICKS:
            g['last_trace'] = tick
            hp = [sum(float(t.hitpoints or 0) for t in obs[0].towers if t.owner == o)
                  / max(sum(float(t.max_hitpoints or 0) for t in obs[0].towers if t.owner == o), 1.) for o in (0, 1)]
            g['trace'].append([tick, round(el[0], 2), round(el[1], 2), round(hp[0], 4), round(hp[1], 4), float(obs[0].time.elixir_multiplier or 1.)])
            fz0 = g['fz'][0]
            for e in obs[0].entities:
                if e.owner is None or e.position is None:
                    continue
                try:
                    t = fz0.unit_index.get(fz0.tz._entity_archetype_id(e))
                except (KeyError, ValueError, AttributeError):
                    continue
                kind = BOARD_KINDS.get(UNIT_KEYS[t - 1]) if t else None
                if kind is None:
                    continue
                hp_, mhp = e.hitpoints, e.max_hitpoints
                g['board'].append([tick, int(e.owner), kind, int(round(float(e.position[0]))), int(round(float(e.position[1]))),
                                   round(float(hp_) / float(mhp), 3) if hp_ is not None and mhp else None])

    def il_decide(g, obs, force):
        batch = g['tz'].tensorize(obs, validate=False)
        start_flag = torch.tensor([g['first']], dtype=torch.bool, device=device)
        with torch.inference_mode():
            model_batch = None
            if graph is not None:
                output = graph.run(il, batch, g['il_state'], start_flag)
            else:
                model_batch = batch.to_model_input(device)
                output = il.act(model_batch, g['il_state'], episode_start=start_flag, validate=False)
            actions = output.actions; forced = False
            if force and int(actions.gate[0]) != GATE_ACT and bool(batch.candidates.mask.any()):
                model_batch = model_batch if model_batch is not None else batch.to_model_input(device)
                context = il.forward(model_batch, g['il_state'], episode_start=start_flag, validate=False)
                actions = il.act_after_preselected_act(model_batch, context, validate=False).actions; forced = True
        g['il_state'] = output.next_state
        return il_finish(g, obs, batch, actions, forced)

    def il_finish(g, obs, batch, actions, forced):
        """Record the opponent's chosen actions in its tensorizer and decode them to native actions (local and GPU paths)."""
        g['tz'].record_action(actions, batch, row=0, validate=False)
        tz = g['tz']
        decoded = decode_action_sequence_v4(actions, batch.candidates, row=0, observation=obs, catalog=tz.catalog, deck=tz.deck,
            card_costs=tz.card_costs, ability_id_by_vocab_id=tz.ability_id_by_vocab_id, horizontal_mirror=tz.perspective.horizontal_mirror,
            hand_slot_permutation=tz.perspective.hand_slot_permutation, config=tz.config, base_latency_ticks=1, validate=False)
        g['first'] = False
        tick = int(obs.tick)
        items = tuple(replace(a, action_id=a.action_id or f"il-{g['row']['game']}-{g['il']}-{tick}-{k}") for k, a in enumerate(decoded.actions))
        return items, int(actions.gate[0]) == GATE_ACT, forced

    def finish(gr, lane, nxt, truncated):
        g = lane['game']; lane['game'] = None
        terminal = nxt[0].terminal; result = [float(v) for v in terminal.result_by_owner]
        final = nxt[0]
        crowns = [int(next(q for q in final.players if q.owner == o).crowns) for o in (0, 1)]
        tower_hp = [sum(float(t.hitpoints or 0) for t in final.towers if t.owner == o)
                    / max(sum(float(t.max_hitpoints or 0) for t in final.towers if t.owner == o), 1.) for o in (0, 1)]
        ticks = [[t for t, _ in g['elix'][o]] for o in (0, 1)]

        def before(o, tick):   # exact elixir of owner o at the last decision frame before `tick`
            k = bisect.bisect_left(ticks[o], tick) - 1
            return round(g['elix'][o][k][1], 2) if k >= 0 else None
        events = sorted(g['events'])
        for ev in events:
            ev += [before(ev[1], ev[0]), before(1 - ev[1], ev[0])]
        live = [[e for t, e in g['elix'][o] if t < GAMEPLAY_END_TICK] for o in (0, 1)]
        elixir = [dict(mean=round(sum(v) / max(len(v), 1), 3), at10=round(sum(x >= 9.999 for x in v) / max(len(v), 1), 4),
                       leaked=round(g['leak'][o], 2)) for o, v in enumerate(live)]
        towers_final = [[[str(t.tower_kind), int(round(float(t.position[0]))), int(round(float(t.position[1]))),
                          int(t.hitpoints or 0), int(t.max_hitpoints or 0)] for t in final.towers if t.owner == o] for o in (0, 1)]
        row = g['row']
        rec = dict(game=row['game'], seed=row['seed'], small_owner=g['small'], engine=gr['spec'].get('engine', gr['spec']['port']),
                   env_id=lane['env_id'], host=gr['spec']['host'], version=row.get('version'), model=g['model'], temperature=g['temp'],
                   opp_version=row.get('opp_version'), opp_model=row.get('opp_model'), opp_temperature=row.get('opp_temperature'),
                   ended=bool(terminal.ended), truncated=truncated, result=result, small_result=result[g['small']],
                   crowns=crowns, tower_hp=[round(x, 4) for x in tower_hp], towers_final=towers_final,
                   plays=[[[t, c, None if xy is None else [round(xy[0], 3), round(xy[1], 3)]] for t, c, xy in g['fz'][o].my_plays] for o in (0, 1)],
                   events=events, ability=[sorted(a) for a in g['ability']], ability_events=g['ability_events'], elixir=elixir, trace=g['trace'], board=g['board'],
                   native=dict(g['native'], actions=[[t, a, g['outcome'].get(a.get('action_id'))] for t, a in g['actions']]),
                   tick=int(nxt[0].tick), acts=g['acts'], first_act_tick=g['first_act'], forced_tick=g['forced'],
                   rejected=g['rejected'], cancelled=g['cancelled'], seconds=time.time() - g['started'],
                   valid=bool(terminal.ended) and not truncated and not any(g['rejected']), utc=time.time())
        with log.open('a') as f:
            f.write(json.dumps(rec, separators=(',', ':')) + '\n')

    for gr in groups:
        for lane in gr['lanes']:
            if gr['queue']:
                start(gr, lane)
        gr['coord'].start_session()
    rounds = 0
    while any(lane['game'] for gr in groups for lane in gr['lanes']):
        t0 = time.monotonic()
        active = [(gr, lane) for gr in groups for lane in gr['lanes'] if lane['game']]
        commands = {}; small_rows = []; il_rows = []

        def il_apply(lane, items, acted, forced):
            g = lane['game']; tick = int(g['obs'][0].tick)
            commands[id(lane)][g['il']] = items
            g['actions'] += [[tick, a.to_dict()] for a in items if a.kind not in (ActionKind.WAIT, ActionKind.WAIT_UNTIL_AFFORDABLE)]
            if acted:
                g['acts'][g['il']] += 1; g['ids'][g['il']].update(a.action_id for a in items); g['fz'][g['il']].acted = True
                if g['first_act'][g['il']] is None: g['first_act'][g['il']] = tick
                if forced: g['forced'][g['il']] = tick
        for gr, lane in active:
            g = lane['game']; obs = g['obs']; tick = int(obs[0].tick)
            t1 = time.monotonic()
            feats = {o: g['fz'][o](obs[o]) for o in (0, 1)}   # both seats: keeps the shared forced-opening rule identical
            observe(g, obs, tick)
            timer['features'] += time.monotonic() - t1
            if tick >= GAMEPLAY_END_TICK:
                commands[id(lane)] = {o: (wait_action(o),) for o in (0, 1)}; continue
            commands[id(lane)] = {}
            if g['tz'] is not None:
                t1 = time.monotonic()
                if il_remote is not None:
                    il_rows.append((lane, g['tz'].tensorize(obs[g['il']], validate=False), bool(feats[g['il']][1]['force'])))
                else:
                    il_apply(lane, *il_decide(g, obs[g['il']], feats[g['il']][1]['force']))
                timer['il'] += time.monotonic() - t1
            for o in g['pol']:
                small_rows.append((lane, o, feats[o]))
        t1 = time.monotonic()
        decided = []          # (lane, seat, (inputs, legality, native), decision)
        if il_remote is not None and il_rows:      # both GPU requests go out before either reply is awaited
            il_remote['send']([2 * lane['slot'] + 1 for lane, _, _ in il_rows], [lane['game']['first'] for lane, _, _ in il_rows],
                              [force for _, _, force in il_rows], [batch for _, batch, _ in il_rows])
        if remote is not None and small_rows:
            seats = [(f[0], f[1], int(lane['game']['pol'][o][2]), 2 * lane['slot'] + o, lane['game']['fresh'][o]) for lane, o, f in small_rows]
            remote['send'](seats, [lane['game']['pol'][o][1] for lane, o, _ in small_rows])
        if il_remote is not None and il_rows:
            reply = il_remote['recv']()
            for j, ((lane, batch, _), forced) in enumerate(zip(il_rows, reply['forced'])):
                g = lane['game']
                il_apply(lane, *il_finish(g, g['obs'][g['il']], batch, reply['actions'].index_select(torch.tensor([j])), bool(forced)))
        timer['il_remote'] += time.monotonic() - t1; t1 = time.monotonic()
        if remote is not None and small_rows:
            for (lane, o, f), d in zip(small_rows, remote['recv'](len(small_rows))):
                lane['game']['fresh'][o] = False; decided.append((lane, o, f, d))
        elif small_rows:
            by_policy = collections.defaultdict(list)
            for lane, o, f in small_rows:
                by_policy[lane['game']['pol'][o]].append((lane, o, f))
            for (path, temp, _v), items_ in by_policy.items():
                x = stack_inputs([f[0] for _, _, f in items_], torch.device('cpu'))
                legal = {k: torch.from_numpy(np.stack([f[1][k] for _, _, f in items_])) for k in ('opt_mask', 'place')}
                legal['force'] = torch.tensor([bool(f[1]['force']) for _, _, f in items_])
                state = tuple(torch.cat([lane['game']['state'][o][i] for lane, o, _ in items_], 1) for i in range(2))
                with torch.no_grad():
                    out_, state = small_model(path).act(x, legal, state, greedy=temp <= 0, temperature=temp)
                for j, (lane, o, f) in enumerate(items_):
                    lane['game']['state'][o] = (state[0][:, j:j + 1], state[1][:, j:j + 1])
                    decided.append((lane, o, f, {k: v[j].item() for k, v in out_.items()}))
        for lane, o, (inputs, legality, native), d in decided:
            g = lane['game']; tick = int(g['obs'][0].tick)
            if d['gate'] == 1:
                opt = int(d['option']); aid = f"zero-{g['row']['game']}-{o}-{tick}"
                action = native_action(o, g['fz'][o], native, opt, int(legality['opt_bin'][opt]), int(d['cell']), aid)
                commands[id(lane)][o] = (action,); g['actions'].append([tick, action.to_dict()])
                g['ids'][o].add(aid); g['acts'][o] += 1; g['fz'][o].acted = True
                if g['first_act'][o] is None: g['first_act'][o] = tick
                if legality['force']: g['forced'][o] = tick
            else:
                commands[id(lane)][o] = (wait_action(o),)
        timer['small'] += time.monotonic() - t1; t1 = time.monotonic()
        stepping = []
        for gr in groups:
            act = [lane for lane in gr['lanes'] if lane['game']]
            if act:
                gr['coord'].begin_round(tuple(lane['env_id'] for lane in act))
                stepping.append((gr, [(lane, pool.submit(lane['env'].step, commands[id(lane)], record_trace=False, policy_features_only=True))
                                      for lane in act]))
        for gr, futures in stepping:
            try:
                for lane, fut in futures:
                    lane['result'] = fut.result()
            except BaseException as error:
                gr['coord'].abort_round(error); raise
            gr['coord'].finish_round()
        timer['native'] += time.monotonic() - t1; t1 = time.monotonic()
        for gr, futures in stepping:
            finished = []
            for lane, _ in futures:
                nxt, _, ended, truncated, _ = lane.pop('result'); g = lane['game']
                for o in (0, 1):
                    for e in nxt[o].events:
                        if e.event_type == 'action_rejected' and e.owner == o and (e.data or {}).get('action_id') in g['ids'][o]:
                            g['ids'][o].discard(e.data['action_id'])
                            status = 'cancelled' if deadline_cancellation(e) else 'rejected'
                            g[status][o] += 1; g['outcome'][e.data['action_id']] = [status, int(e.tick)]
                g['obs'] = nxt
                if ended['__all__'] or truncated['__all__']:
                    finished.append((lane, nxt, bool(truncated['__all__'])))
            if finished:
                gr['coord'].close_session()
                for lane, nxt, truncated in finished:
                    finish(gr, lane, nxt, truncated)
                    if gr['queue']:
                        start(gr, lane)
                if any(lane['game'] for lane in gr['lanes']):
                    gr['coord'].start_session()
        timer['finish'] += time.monotonic() - t1
        timer['round'] += time.monotonic() - t0; timer['seat_rounds'] += len(active); rounds += 1
    atomic(out / 'games' / f'w{index:03d}.timing.json', dict(rounds=rounds, **timer))
    for gr in groups:
        for lane in gr['lanes']:
            try:
                lane['native'].close_transport()
            except Exception:
                pass


def node(out, host):
    """One process per engine host: a worker per local engine."""
    plan = json.loads((out / 'plan.json').read_text())
    meta = json.loads((out / 'config.json').read_text())
    mine = [(e, rows) for e, rows in plan if e['host'] == host]
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=_worker_entry, args=(e['engine'], [(e, rows)], out, meta['small_checkpoint'], meta['small_mode'], 'cpu'))
             for e, rows in mine]
    for p in procs: p.start()
    for p in procs: p.join()
    return 0 if all(p.exitcode == 0 for p in procs) else 1


def _worker_entry(*args):
    try:
        worker(*args)
    except BaseException:
        out = args[2]
        with (out / 'games' / f'w{args[0]:03d}.failure').open('w') as f:
            f.write(traceback.format_exc())
        raise


def pool_engines():
    res = load_engines()
    hosts = collections.defaultdict(list)
    for i, spec in enumerate(res['collectors']):
        hosts[spec['host']].append(dict(spec, engine=i))
    return res, [e for group in itertools.zip_longest(*hosts.values()) for e in group if e is not None]   # interleave hosts


def assign(rows, engines, contiguous):
    """Round-robin (one game per engine for small evals), or contiguous chunks so an engine sees few snapshots (sweeps)."""
    if not contiguous:
        queued = [(e, []) for e in engines]
        for k, row in enumerate(rows):
            queued[k % len(engines)][1].append(row)
    else:
        n = len(engines); base, extra = divmod(len(rows), n); queued = []; k = 0
        for i, e in enumerate(engines):
            take = base + (i < extra); queued.append((e, rows[k:k + take])); k += take
    return [q for q in queued if q[1]]


def run_plan(out, rows, meta, contiguous=False, max_engines=None):
    """Plays `rows` on the engines (8 games at once per engine), replays games of failed workers
    on other engines (up to 3 attempts). Writes config.json / plan.json; returns run statistics."""
    res, engines = pool_engines()
    engines = engines[:max_engines] if max_engines else engines
    queued = assign(rows, engines, contiguous)
    meta = dict(meta, engines=len(queued), games_per_engine=sorted({len(q[1]) for q in queued}),
                slots_per_engine=SLOTS_PER_ENGINE, scheduled_games=len(rows), il_device='cpu', small_device='cpu')
    atomic(out / 'config.json', meta)
    started = time.time(); (out / 'logs').mkdir(exist_ok=True); bad = set(); retried = []
    for attempt in range(3):   # games of failed workers are replayed on engines that have not failed
        atomic(out / 'plan.json', queued)
        steps = []
        for host in sorted({e['host'] for e, _ in queued}):
            log = (out / 'logs' / f'{host}-{attempt}.log').open('w')
            steps.append(spawn_on(host, [sys.executable, str(Path(__file__).resolve()), 'node', '--out', str(out), '--host', host], log))
        for step in steps: step.wait()
        failures = sorted((out / 'games').glob('*.failure'))
        done = {json.loads(l)['game'] for f in (out / 'games').glob('w*.jsonl') for l in f.read_text().splitlines()}
        missing = [row for row in rows if row['game'] not in done]
        if failures:
            failed_dir = out / 'games' / f'failed-{attempt}'; failed_dir.mkdir()
            for f in failures: f.rename(failed_dir / f.name)
            bad.update(int(f.stem[1:]) for f in failures)
        if not missing:
            break
        spare = [e for e in engines if e['engine'] not in bad]
        retried.append(dict(attempt=attempt, failed_engines=sorted(bad), replayed_games=len(missing)))
        queued = assign(missing, spare, contiguous)
    else:
        raise RuntimeError(f'eval workers kept failing after retries: {retried}; see {out}/games/failed-*')
    return dict(wall_seconds=time.time() - started, retries=retried)


def protocol(opponent, il_checkpoint):
    return dict(opponent=Path(opponent).stem, il_checkpoint=str(il_checkpoint), il_sha256=sha(il_checkpoint), il_mode='argmax',
                delay_policy=dict(small='earliest', il='earliest'), allowed_delay_offsets_ms=dict(small=DELAYS_MS, il=DELAYS_MS),
                delay_ceiling_ms=dict(small=200, il=200), forced_opening='all-owners', force_after_seconds=FORCE_AFTER_SECONDS,
                execution_jitter_ticks=0, battle_level=11, deck='3.0 X-Bow mirror')


def summarize(out, meta):
    rows = [json.loads(l) for f in sorted((out / 'games').glob('w*.jsonl')) for l in f.read_text().splitlines()]
    result = dict(meta, **stats(rows), utc=time.time())
    atomic(out / 'result.json', result)
    return result


def stats(rows):
    """Result statistics of one set of game records (all games of one snapshot / temperature)."""
    valid = [r for r in rows if r['valid']]
    wins = sum(r['small_result'] > 0 for r in valid); losses = sum(r['small_result'] < 0 for r in valid); draws = len(valid) - wins - losses
    score = (wins + .5 * draws) / max(len(valid), 1)
    se = math.sqrt(max(score * (1 - score), 1e-9) / max(len(valid), 1))
    side = lambda r, who: r['small_owner'] if who == 'small' else 1 - r['small_owner']
    seat = {str(k): dict(games=len(v), score=(sum(r['small_result'] > 0 for r in v) + .5 * sum(r['small_result'] == 0 for r in v)) / max(len(v), 1))
            for k, v in ((k, [r for r in valid if r['small_owner'] == k]) for k in (0, 1))}
    per_side = {who: dict(acts_per_game=sum(r['acts'][side(r, who)] for r in valid) / max(len(valid), 1),
                          forced_games=sum(r['forced_tick'][side(r, who)] is not None for r in valid),
                          never_played=sum(r['first_act_tick'][side(r, who)] is None for r in valid),
                          rejected=sum(r['rejected'][side(r, who)] for r in rows), cancelled=sum(r['cancelled'][side(r, who)] for r in rows))
                for who in ('small', 'il')}
    crowns = lambda r, who: r['crowns'][side(r, who)]
    hp = lambda r, who: r['tower_hp'][side(r, who)]
    margin = dict(small_crowns=sum(crowns(r, 'small') for r in valid) / max(len(valid), 1),
                  il_crowns=sum(crowns(r, 'il') for r in valid) / max(len(valid), 1),
                  small_tower_hp=sum(hp(r, 'small') for r in valid) / max(len(valid), 1),
                  il_tower_hp=sum(hp(r, 'il') for r in valid) / max(len(valid), 1))
    return dict(games=len(rows), valid=len(valid), wins=wins, losses=losses, draws=draws, score_rate=score, score_se=se, margin=margin,
                by_small_seat=seat, per_side=per_side, mean_end_tick=sum(r['tick'] for r in valid) / max(len(valid), 1))


def eval_dir(run, version, small_mode, opponent):
    return run / 'evals' / f'v{version}-{small_mode}-{Path(opponent).stem}'


def evaluate(run, version, small_mode, games, opponent):
    settings = json.loads((run / 'settings.json').read_text())
    source = run / 'weights/history' / f'v{version}.pt'
    il_checkpoint = Path(opponent).resolve()
    out = eval_dir(run, version, small_mode, opponent)
    if (out / 'result.json').exists():
        done = json.loads((out / 'result.json').read_text())
        if not done.get('skipped'):   # loop placeholders for versions it passed over can still be run explicitly
            return done
    if out.exists():
        shutil.rmtree(out)
    (out / 'games').mkdir(parents=True); (out / 'frozen').mkdir()
    small_path = out / 'frozen' / source.name; shutil.copy2(source, small_path)
    atomic(out / 'small-model.json', settings['model'])
    rows = [dict(r, version=version) for r in schedule(games)]
    meta = dict(schema='zero-vs-opponent.v1', run=str(run), version=version, small_checkpoint=str(small_path), small_sha256=sha(small_path),
                small_mode=small_mode, **protocol(opponent, il_checkpoint), schedule_seed=SEED, schedule='paired')
    info = run_plan(out, rows, meta)
    result = summarize(out, dict(json.loads((out / 'config.json').read_text()), **info))
    with (run / 'evals.jsonl').open('a') as f:
        f.write(json.dumps({k: result[k] for k in ('version', 'small_mode', 'opponent', 'games', 'valid', 'wins', 'losses', 'draws', 'score_rate',
                                                     'score_se', 'margin', 'mean_end_tick', 'wall_seconds', 'utc')}) + '\n')
    return result


def parse_versions(text):
    out = []
    for part in text.split(','):
        if ':' in part:
            lo, hi, step = (int(v) for v in part.split(':')); out += list(range(lo, hi + 1, step))
        elif part:
            out.append(int(part))
    return out


def sweep(run, name, versions, temps, games, opponent, kind, max_engines=None, inference='local', il_inference='local'):
    """Every (snapshot, temperature) plays the same `games` scheduled seeds against `opponent`, all in one plan."""
    settings = json.loads((run / 'settings.json').read_text())
    out = run / 'sweeps' / name
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    if out.exists():
        shutil.rmtree(out)
    (out / 'games').mkdir(parents=True); (out / 'frozen').mkdir()
    atomic(out / 'small-model.json', settings['model'])
    models = {}
    for v in versions:
        dst = out / 'frozen' / f'v{v}.pt'; shutil.copy2(run / 'weights/history' / f'v{v}.pt', dst)
        models[str(v)] = dict(path=str(dst), sha256=sha(dst))
    base = schedule(games, kind, SWEEP_SEED)
    rows = []
    for v in versions:
        for t in temps:
            rows += [dict(r, game=len(rows) + k, version=v, temperature=t, model=models[str(v)]['path']) for k, r in enumerate(base)]
    first = models[str(versions[0])]['path']
    meta = dict(schema='zero-sweep.v1', run=str(run), name=name, versions=versions, temperatures=temps, games_per_policy=games,
                models=models, small_checkpoint=first, small_mode='sample', schedule=kind, schedule_seed=SWEEP_SEED,
                seeds=[r['seed'] for r in base], inference=inference, server_registry=str(run / 'servers'),
                il_inference=il_inference, il_server_registry=str(run / 'il-servers'),
                **protocol(opponent, Path(opponent).resolve()))
    info = run_plan(out, rows, meta, contiguous=True, max_engines=max_engines)
    atomic(out / 'run.json', info)
    return summarize_sweep(run, name)


CARD_NAMES = {26000000: 'Knight', 26000001: 'Archers', 26000010: 'Skeletons', 26000084: 'ESpirit',
              27000006: 'Tesla', 27000008: 'XBow', 28000000: 'Fireball', 28000011: 'Log'}


def side_profile(games, who):
    """Per-game averages of one side ('small' or 'il') over game records: plays per card, Hero ability uses, elixir held,
    share of frames at 10, elixir right before each card, mean landing point (own frame, 0..1) and left-lane share."""
    n = max(len(games), 1); plays = collections.Counter(); el_before = collections.defaultdict(list); pos = collections.defaultdict(list)
    ability = 0; mean_el = 0.; at10 = 0.
    for r in games:
        o = who(r) if callable(who) else r['small_owner'] if who == 'small' else 1 - r['small_owner']
        ability += len(r['ability'][o]); mean_el += r['elixir'][o]['mean']; at10 += r['elixir'][o]['at10']
        for t, owner, card, x, y, cost, e_own, e_opp in r['events']:
            if owner != o:
                continue
            name = CARD_NAMES.get(card, str(card)); plays[name] += 1
            if e_own is not None: el_before[name].append(e_own)
            if x is not None:
                fx, fy = (x / 18000., y / 32000.) if o == 0 else (1 - x / 18000., 1 - y / 32000.)
                pos[name].append((fx, fy))
    mean = lambda v: sum(v) / len(v) if v else None
    return dict(plays_per_game={k: v / n for k, v in sorted(plays.items())}, ability_per_game=ability / n,
                elixir_mean=mean_el / n, elixir_at10=at10 / n,
                elixir_before_play={k: mean(v) for k, v in sorted(el_before.items())},
                landing_mean={k: [round(mean([p[0] for p in v]), 3), round(mean([p[1] for p in v]), 3)] for k, v in sorted(pos.items())},
                left_lane_share={k: round(sum(p[0] < .5 for p in v) / len(v), 3) for k, v in sorted(pos.items())})


def summarize_sweep(run, name):
    out = run / 'sweeps' / name; meta = json.loads((out / 'config.json').read_text())
    by = collections.defaultdict(list)
    for f in sorted((out / 'games').glob('w*.jsonl')):
        for l in f.read_text().splitlines():
            r = json.loads(l); by[r['version'], r['temperature']].append(r)
    policies = []
    for (v, t), rows in sorted(by.items()):
        valid = [r for r in rows if r['valid']]
        s = stats(rows)
        policies.append(dict(version=v, temperature=t, **s, mean_game_seconds=s['mean_end_tick'] / 20.,
                             tower_hp_margin=s['margin']['small_tower_hp'] - s['margin']['il_tower_hp'],
                             small=side_profile(valid, 'small'), il=side_profile(valid, 'il')))
    result = dict(meta, policies=policies, run_info=json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else None,
                  utc=time.time())
    atomic(out / 'result.json', result)
    cols = ['version', 'temperature', 'valid', 'wins', 'draws', 'losses', 'score_rate', 'score_se', 'tower_hp_margin', 'mean_game_seconds']
    cards = sorted(CARD_NAMES.values())
    with (out / 'summary.csv').open('w') as f:
        f.write(','.join(cols + [f'plays_{c}' for c in cards] + ['ability', 'elixir_mean', 'elixir_at10']) + '\n')
        for p in policies:
            f.write(','.join([str(round(p[c], 4) if isinstance(p[c], float) else p[c]) for c in cols]
                             + [str(round(p['small']['plays_per_game'].get(c, 0.), 3)) for c in cards]
                             + [str(round(p['small'][k], 4)) for k in ('ability_per_game', 'elixir_mean', 'elixir_at10')]) + '\n')
    return result


def top_versions(run, name, k):
    """The k best snapshots of an IL sweep: score rate, then tower-hp margin, then crowns margin (ties at 100 % are common)."""
    pol = json.loads((run / 'sweeps' / name / 'result.json').read_text())['policies']
    key = lambda p: (p['score_rate'], p['tower_hp_margin'], p['margin']['small_crowns'] - p['margin']['il_crowns'], p['version'])
    return [p['version'] for p in sorted(pol, key=key, reverse=True)[:k]]


def roundrobin(run, name, policies, seeds_per_pair, max_engines=None, inference='local'):
    """Every pair of zero policies (snapshot, temperature) plays `seeds_per_pair` seeds x both seats; each pair gets its own seeds
    (one shared seed would give every game the same deal, and argmax pairs would rank on that one opening)."""
    out = run / 'sweeps' / name
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    if out.exists():
        shutil.rmtree(out)
    (out / 'games').mkdir(parents=True); (out / 'frozen').mkdir()
    atomic(out / 'small-model.json', json.loads((run / 'settings.json').read_text())['model'])
    models = {}
    for v in sorted({v for v, _ in policies}):
        dst = out / 'frozen' / f'v{v}.pt'; shutil.copy2(run / 'weights/history' / f'v{v}.pt', dst)
        models[str(v)] = dict(path=str(dst), sha256=sha(dst))
    pairs = [(a, b) for i, a in enumerate(policies) for b in policies[i + 1:]]
    base = schedule(2 * seeds_per_pair * len(pairs), 'paired', SWEEP_SEED + 1)
    rows = []
    for n, ((va, ta), (vb, tb)) in enumerate(pairs):
        rows += [dict(r, game=len(rows) + k, version=va, temperature=ta, model=models[str(va)]['path'],
                      opp_version=vb, opp_temperature=tb, opp_model=models[str(vb)]['path'])
                 for k, r in enumerate(base[2 * seeds_per_pair * n: 2 * seeds_per_pair * (n + 1)])]
    meta = dict(schema='zero-roundrobin.v1', run=str(run), name=name, policies=[list(p) for p in policies], seeds_per_pair=seeds_per_pair,
                models=models, small_checkpoint=models[str(policies[0][0])]['path'], small_mode='sample', schedule='paired',
                schedule_seed=SWEEP_SEED + 1, seeds=sorted({r['seed'] for r in base}), opponent='zero-roundrobin', il_checkpoint=None,
                delay_policy=dict(small='earliest', il='earliest'), allowed_delay_offsets_ms=dict(small=DELAYS_MS, il=DELAYS_MS),
                delay_ceiling_ms=dict(small=200, il=200), forced_opening='all-owners', force_after_seconds=FORCE_AFTER_SECONDS,
                execution_jitter_ticks=0, battle_level=11, deck='3.0 X-Bow mirror', inference=inference, server_registry=str(run / 'servers'))
    info = run_plan(out, rows, meta, contiguous=True, max_engines=max_engines)
    atomic(out / 'run.json', info)
    return summarize_roundrobin(run, name)


def bradley_terry(games, keys, iters=2000):
    """Log-strengths maximising sum s*log sig(ri-rj) + (1-s)*log sig(rj-ri) (draw = half), mean 0; plain gradient ascent."""
    idx = {k: i for i, k in enumerate(keys)}; r = [0.] * len(keys); n = len(keys); lam = .01   # weak N(0, 10^2) prior: finite under sweeps
    pairs = [(idx[a], idx[b], s) for a, b, s in games]
    for _ in range(iters):
        grad = [-lam * x for x in r]; hess = [lam] * n
        for i, j, s in pairs:
            p = 1. / (1. + math.exp(r[j] - r[i])); w = p * (1. - p)
            grad[i] += s - p; grad[j] -= s - p; hess[i] += w; hess[j] += w
        r = [x + .8 * gx / hx for x, gx, hx in zip(r, grad, hess)]      # damped diagonal Newton
        m = sum(r) / n; r = [x - m for x in r]
    return r


def summarize_roundrobin(run, name, boots=200):
    out = run / 'sweeps' / name; meta = json.loads((out / 'config.json').read_text())
    recs = [json.loads(l) for f in sorted((out / 'games').glob('w*.jsonl')) for l in f.read_text().splitlines()]
    valid = [r for r in recs if r['valid']]
    A = lambda r: (r['version'], r['temperature']); B = lambda r: (r['opp_version'], r['opp_temperature'])
    games = [(A(r), B(r), 1. if r['small_result'] > 0 else 0. if r['small_result'] < 0 else .5) for r in valid]
    keys = [tuple(p) for p in meta['policies']]
    elo = 400. / math.log(10.)
    r = bradley_terry(games, keys)
    rng = random.Random(0); samples = []; first = collections.Counter()
    for _ in range(boots):
        g = [games[rng.randrange(len(games))] for _ in games]; rb = bradley_terry(g, keys, iters=400); samples.append(rb)
        first[keys[max(range(len(keys)), key=lambda i: rb[i])]] += 1
    rows = []
    for i, k in enumerate(keys):
        mine = [g for g in games if k in (g[0], g[1])]
        score = sum(s if g[0] == k else 1 - s for g in mine for s in (g[2],)) / max(len(mine), 1)
        boot = sorted(s[i] for s in samples)
        pol_games = [rec for rec in valid if A(rec) == k or B(rec) == k]
        rows.append(dict(version=k[0], temperature=k[1], elo=round(elo * r[i], 1), elo_ci95=[round(elo * boot[int(.025 * boots)], 1),
                         round(elo * boot[int(.975 * boots) - 1], 1)], p_best=first[k] / boots, games=len(mine), score_rate=round(score, 4),
                         profile=side_profile(pol_games, lambda rec, k=k: rec['small_owner'] if A(rec) == k else 1 - rec['small_owner'])))
    rows.sort(key=lambda x: -x['elo'])
    by_version = collections.defaultdict(list); by_temp = collections.defaultdict(list)
    for x in rows:
        by_version[x['version']].append(x['elo']); by_temp[x['temperature']].append(x['elo'])
    result = dict(meta, ranking=rows, mean_elo_by_version={str(k): round(sum(v) / len(v), 1) for k, v in by_version.items()},
                  mean_elo_by_temperature={str(k): round(sum(v) / len(v), 1) for k, v in by_temp.items()},
                  games=len(recs), valid=len(valid), run_info=json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else None,
                  utc=time.time())
    atomic(out / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['once', 'loop', 'node', 'sweep', 'summarize-sweep', 'roundrobin', 'summarize-roundrobin'])
    p.add_argument('--top-from', help='roundrobin: take the --top best snapshots of this IL sweep')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--seeds-per-pair', type=int, default=1, help='roundrobin: seeds per pair, each played from both seats')
    p.add_argument('--run', type=Path)
    p.add_argument('--out', type=Path); p.add_argument('--host')
    p.add_argument('--version', type=int, help='history snapshot; default: newest')
    p.add_argument('--small-mode', choices=['sample', 'argmax'], default='sample')
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--every', type=int, default=50)
    p.add_argument('--opponent', help='V4 opponent checkpoint (once, loop, sweep)')
    p.add_argument('--name', help='sweep directory name under RUN/sweeps')
    p.add_argument('--versions', help="sweep snapshots, e.g. '10:1500:10' or '1460,1480,1500'")
    p.add_argument('--temps', default='1', help="sweep temperatures, comma separated; 0 = argmax")
    p.add_argument('--schedule', choices=['paired', 'single'], default='single')
    p.add_argument('--max-engines', type=int, help='sweep: use only the first N engines (tests)')
    p.add_argument('--il-inference', choices=['local', 'server'], default='local', help='sweep: IL opponent on this CPU or on RUN/il-servers')
    p.add_argument('--inference', choices=['local', 'server'], default='local',
                   help='sweep/roundrobin: zero seats on the CPU worker, or on the GPU inference servers registered in RUN/servers')
    a = p.parse_args()
    if a.action == 'node':
        sys.exit(node(a.out, a.host))
    run = a.run.resolve()
    if a.action in ('once', 'loop', 'sweep') and not a.opponent:
        p.error('--opponent PATH is required')
    if a.action == 'sweep':
        r = sweep(run, a.name, parse_versions(a.versions), [float(t) for t in a.temps.split(',')], a.games, a.opponent, a.schedule, a.max_engines, a.inference, a.il_inference)
        print(json.dumps(dict(name=a.name, policies=len(r['policies']), run_info=r.get('run_info'))), flush=True)
        return
    if a.action == 'summarize-sweep':
        r = summarize_sweep(run, a.name); print(json.dumps(dict(name=a.name, policies=len(r['policies']))), flush=True)
        return
    if a.action in ('roundrobin', 'summarize-roundrobin'):
        if a.action == 'roundrobin':
            versions = top_versions(run, a.top_from, a.top) if a.top_from else parse_versions(a.versions)
            policies = [(v, float(t)) for v in versions for t in a.temps.split(',')]
            r = roundrobin(run, a.name, policies, a.seeds_per_pair, a.max_engines, a.inference)
        else:
            r = summarize_roundrobin(run, a.name)
        print(json.dumps(dict(name=a.name, games=r['games'], top=r['ranking'][:5]), default=str)[:3000], flush=True)
        return
    snapshots = lambda: sorted(int(x.stem[1:]) for x in (run / 'weights/history').glob('v*.pt'))
    if a.action == 'once':
        r = evaluate(run, a.version or snapshots()[-1], a.small_mode, a.games, a.opponent)
        print(json.dumps({k: r[k] for k in ('version', 'small_mode', 'opponent', 'valid', 'wins', 'losses', 'draws', 'score_rate', 'margin', 'per_side')}), flush=True)
        return
    while not (run / 'STOP').exists():
        todo = [v for v in snapshots() if v % a.every == 0 and not (eval_dir(run, v, a.small_mode, a.opponent) / 'result.json').exists()]
        if todo:
            try:
                r = evaluate(run, todo[-1], a.small_mode, a.games, a.opponent)   # newest first; older gaps are skipped
            except RuntimeError as error:
                print(f'v{todo[-1]} eval failed, retrying in 5 min:', str(error)[:2000], flush=True)
                time.sleep(300); continue
            print(json.dumps({k: r[k] for k in ('version', 'opponent', 'valid', 'score_rate', 'margin')}), flush=True)
            for v in todo[:-1]:
                d = eval_dir(run, v, a.small_mode, a.opponent); d.mkdir(parents=True, exist_ok=True)
                atomic(d / 'result.json', dict(version=v, skipped=True))
        else:
            time.sleep(60)


if __name__ == '__main__':
    main()
