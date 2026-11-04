"""One process per native engine: 8 resident games, remote GPU inference, trajectories to the run inbox.

Opponents: half self-play, half current-vs-history by PFSP. Decisions for all seats of a frame go
to one inference server (firstlight_zero.server) in a single request; the server keeps the LSTM states.
"""
from pathlib import Path
import argparse, json, os, queue, random, signal, socket, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import numpy as np

from native_runner.cr_native_env import ResidentNativeClashEnv
from native_runner.resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1
from native_runner.training.v4.ppo_runtime import ppo_environment_config_v4, build_ppo_environment_v4, episode_for_matchup_v4
from native_runner.training.v4.deck_pool import DeckPoolManifestV4
from native_runner.training.v4.matchmaking import ScheduledMatchupV4, POLICY_CURRENT
from firstlight_zero.reward import RewardAdapter
from firstlight_zero.reward import deadline_cancellation

from firstlight_zero.actions import native_action, wait_action
from firstlight_zero.spec import XBOW_DECK_ID, POOL, BATTLE_LEVEL, GAMEPLAY_END_TICK, CARD_INDEX
from firstlight_zero.features import Featurizer2
from firstlight_zero.legality import _player
from firstlight_zero.store import SeatStore
from firstlight_zero.batch import pack_request
from firstlight_zero.server import send_msg, recv_msg
from firstlight_zero.hosts import is_local, load_engines, pin

RESET = dict(render_mode='headless', decision_ticks=5, event_driven_decisions=False, reuse_runner_attestation=True)


def emit(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


class Roster:
    """Current version, history snapshots and PFSP scores (files written by the learner)."""

    def __init__(self, run):
        self.run = Path(run); self.version = -1; self.history = []; self.pfsp = {}; self.refresh()

    def refresh(self):
        try:
            self.version = json.loads((self.run / 'weights/latest.json').read_text())['version']
        except (OSError, ValueError):
            pass
        self.history = sorted(int(p.stem[1:]) for p in (self.run / 'weights/history').glob('v*.pt'))
        try:
            self.pfsp = {int(v): x for v, x in json.loads((self.run / 'weights/pfsp.json').read_text())['stats'].items()}
        except (OSError, ValueError, KeyError):
            pass

    def opponent_weights(self, power):
        weights = []
        for v in self.history:
            n, score = self.pfsp.get(v, (0, .5))
            weights.append((1 - (n * score + 1) / (n + 2)) ** power)
        return weights


class Client:
    def __init__(self, run, index):
        servers = sorted((Path(run) / 'servers').glob('*.json'))
        if not servers:
            raise RuntimeError('no inference server registered')
        s = json.loads(servers[index % len(servers)].read_text())
        self.sock = socket.create_connection((s['host'], s['port']), timeout=120.)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client = index; self.server = f"{s['host']}:{s['port']}"

    def act(self, seats):
        """seats: list of (inputs, legality, model_key, state_key, start) -> (version, [decision dict per seat])."""
        send_msg(self.sock, pack_request(self.client, seats))
        reply = recv_msg(self.sock)
        if 'error' in reply:
            raise RuntimeError('inference server: ' + reply['error'])
        keys = ('gate', 'option', 'cell', 'gate_free', 'logp', 'value')
        return reply['version'], [{k: reply[k][j].item() for k in keys} for j in range(len(seats))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    run = a.run; s = json.loads((run / 'settings.json').read_text())
    spec = load_engines(s['engines'])['collectors'][a.index]
    assert is_local(spec['host']), spec
    pin([spec.get('cpu')])
    log = run / 'collectors' / f'c{a.index:03d}.jsonl'; log.parent.mkdir(exist_ok=True)
    inbox = run / 'inbox'; inbox.mkdir(exist_ok=True)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    done = lambda: stop[0] or (run / 'STOP').exists()

    seed = int(s['seed']) + 7919 * a.index + int(time.time()) % 100000
    rng = random.Random(seed)
    deck = next(d for d in DeckPoolManifestV4.load(POOL).decks if d.deck_id == XBOW_DECK_ID)
    roster = Roster(run); client = Client(run, a.index)

    slots = list(spec['env_ids'])
    natives = {i: ResidentNativeClashEnv(i, '127.0.0.1', spec['port'], timeout=120.) for i in slots}
    coord = ResidentBatchCoordinatorV1(tuple(natives.values()), timeout=120., training_capture=True,
                                       compressed_json=False, validate_json_capture=False)
    cfg = ppo_environment_config_v4(gamma_per_decision=s['gamma'], shaping_beta=0.)
    envs = {i: build_ppo_environment_v4(ResidentBatchNativeProxyV1(n, coord), cfg) for i, n in natives.items()}
    for env in envs.values():
        env.simulated_card_delay_jitter_ticks = int(s['jitter_ticks']); env.prepare_runner_attestation()
    pool = ThreadPoolExecutor(len(slots))
    games = {}; serial = [0]; timer = dict(features=0., policy=0., native=0., reward=0., reset=0., write=0.)
    # GPFS occasionally stalls a 1 MB write for seconds (minutes on some nodes); do the trajectory writes off the game loop.
    writes = queue.Queue(); write_errors = []

    def writer():
        while True:
            item = writes.get()
            if item is None: return
            path, store, meta, record = item
            try:
                t0 = time.monotonic(); store.write(path, meta); record['write_s'] = time.monotonic() - t0
            except Exception:
                write_errors.append(traceback.format_exc()[-800:]); record['write_failed'] = True
            emit(log, record)
    threading.Thread(target=writer, daemon=True).start()
    frames = 0; last_log = time.monotonic(); completed = 0

    def reset(slot):
        started = time.monotonic(); serial[0] += 1
        history = roster.history
        if history and rng.random() >= s['self_play_fraction']:
            learner = rng.randrange(2)
            opponent = rng.choice(history) if rng.random() < s.get('pfsp_uniform', 0.) else \
                rng.choices(history, roster.opponent_weights(s.get('pfsp_power', 2.)))[0]
            roles = {learner: 'current', 1 - learner: opponent}
        else:
            roles = {0: 'current', 1: 'current'}
        m = ScheduledMatchupV4(sequence=serial[0], category='xbow-mirror', policy_matchup=POLICY_CURRENT, deck0=deck, deck1=deck,
                               current_owners=(0, 1), battle_level=BATTLE_LEVEL, seed=rng.randrange(1, 2**31 - 1))
        layouts = m.runtime_layouts(); episode = episode_for_matchup_v4(m, cfg)
        obs, _ = envs[slot].reset(episode, options=RESET)
        games[slot] = dict(obs=obs, roles=roles, id=f'{a.index}-{slot}-{serial[0]}-{seed % 9973}',
            fz={o: Featurizer2(o, episode, obs[o], layouts[o].card_ids) for o in (0, 1)}, fresh={0: True, 1: True},
            seats={o: SeatStore() for o in (0, 1) if roles[o] == 'current'},
            reward=RewardAdapter(envs[slot], obs, s['reward']), ids={}, rejected=[0, 0], cancelled=[0, 0], steps=0,
            first_act=[None, None], forced=[None, None], acts=[0, 0], started=time.time())
        timer['reset'] += time.monotonic() - started

    def finish(slot, nxt, truncated):
        g = games.pop(slot); started = time.monotonic()
        terminal = nxt[0].terminal
        result = [float(v) for v in terminal.result_by_owner]
        valid = bool(terminal.ended) and not truncated and not any(g['rejected'])
        report = g['reward'].report()
        totals = {k: [float(x) for x in v] for k, v in report.items()
                  if k.endswith('_total') or k in ('terminal_outcome_reward', 'tower_shaping_reward')}
        record = dict(event='game', game=g['id'], valid=valid, result=result, tick=int(nxt[0].tick),
                      roles={str(o): r for o, r in g['roles'].items()}, steps=g['steps'], rejected=g['rejected'], cancelled=g['cancelled'],
                      first_act=g['first_act'], forced=g['forced'], acts=g['acts'], version=roster.version,
                      plays=[[[t, c] for t, c, _ in g['fz'][o].my_plays] for o in (0, 1)],
                      other_units=sum(f.other_units for f in g['fz'].values()), utc=time.time())
        if valid and g['seats']:
            for o, store in g['seats'].items():
                plays = [0] * 8
                for _, card, _ in g['fz'][o].my_plays:
                    if card in CARD_INDEX: plays[CARD_INDEX[card]] += 1
                meta = dict(game=g['id'], owner=o, outcome=result[o], opponent=g['roles'][1 - o], collector=a.index,
                            tick=int(nxt[0].tick), acts=g['acts'][o], plays_by_card=plays, crowns=int(_player(nxt[o], o).crowns))
                writes.put((inbox / f"{g['id']}-{o}.npz", store, meta, record))
        else:
            emit(log, record)
        timer['write'] += time.monotonic() - started

    try:
        for slot in slots: reset(slot)
        coord.start_session()
        while not done():
            if frames % 20 == 0:
                roster.refresh()
            started = time.monotonic()
            pending = []   # (slot, owner, inputs, legality, native)
            for slot, g in games.items():
                if g['obs'][0].tick >= GAMEPLAY_END_TICK:
                    continue
                for o in (0, 1):
                    inputs, legality, native = g['fz'][o](g['obs'][o])
                    pending.append((slot, o, inputs, legality, native))
            timer['features'] += time.monotonic() - started; started = time.monotonic()
            decisions = {}; version = roster.version
            if pending:
                seats = [(inputs, legality, games[slot]['roles'][o], slot * 2 + o, games[slot]['fresh'][o])
                         for slot, o, inputs, legality, _ in pending]
                version, outs = client.act(seats)
                for (slot, o, *_), d in zip(pending, outs):
                    decisions[slot, o] = d; games[slot]['fresh'][o] = False
            timer['policy'] += time.monotonic() - started
            commands = {slot: {o: (wait_action(o),) for o in (0, 1)} for slot in games}
            for slot, o, inputs, legality, native in pending:
                g = games[slot]; d = decisions[slot, o]
                if d['gate'] == 1:
                    opt = int(d['option']); aid = f"zero-{g['id']}-{o}-{g['steps']}"
                    commands[slot][o] = (native_action(o, g['fz'][o], native, opt, int(legality['opt_bin'][opt]), int(d['cell']), aid),)
                    g['ids'][aid] = True; g['acts'][o] += 1; g['fz'][o].acted = True
                    if g['first_act'][o] is None: g['first_act'][o] = int(g['obs'][o].tick)
                    if legality['force']: g['forced'][o] = int(g['obs'][o].tick)
                if o in g['seats']:
                    g['seats'][o].append(inputs, legality, d, version)
            started = time.monotonic()
            coord.begin_round(tuple(commands))
            futures = {slot: pool.submit(envs[slot].step, c, record_trace=False, policy_features_only=True) for slot, c in commands.items()}
            try:
                results = {slot: f.result() for slot, f in futures.items()}
            except BaseException as error:
                coord.abort_round(error); raise
            coord.finish_round(); timer['native'] += time.monotonic() - started; started = time.monotonic()
            finished = []
            for slot, (nxt, _, ended, truncated, _) in results.items():
                g = games[slot]; active = g['obs'][0].tick < GAMEPLAY_END_TICK
                values, replace_last = g['reward'].transition(g['obs'], nxt, active)
                if active:
                    g['steps'] += 1
                    for o, store in g['seats'].items(): store.reward(values[o])
                elif replace_last:
                    for o, store in g['seats'].items(): store.reward(values[o], replace=True)
                for o in (0, 1):
                    for e in nxt[o].events:
                        if e.event_type == 'action_rejected' and e.owner == o and g['ids'].pop((e.data or {}).get('action_id'), None):
                            g['cancelled' if deadline_cancellation(e) else 'rejected'][o] += 1
                g['obs'] = nxt
                if ended['__all__'] or truncated['__all__']:
                    finished.append((slot, nxt, bool(truncated['__all__'])))
            timer['reward'] += time.monotonic() - started
            if finished:
                coord.close_session()
                for slot, nxt, truncated in finished:
                    finish(slot, nxt, truncated); completed += 1
                    if not done(): reset(slot)
                coord.start_session()
            frames += 1
            if time.monotonic() - last_log > 60:
                emit(log, dict(event='timing', frames=frames, completed=completed, version=roster.version, server=client.server,
                               write_queue=writes.qsize(), write_errors=len(write_errors), utc=time.time(), **timer))
                last_log = time.monotonic(); frames = 0; completed = 0
                for k in timer: timer[k] = 0.
    except Exception:
        if not done():
            emit(log, dict(event='failure', error=traceback.format_exc(), utc=time.time())); raise
    finally:
        writes.put(None)
        try: coord.abort_round(InterruptedError('collector exit')); coord.close_session()
        except Exception: pass
        pool.shutdown(wait=False, cancel_futures=True)
        for native in natives.values():
            native.close_transport(); native.timeout = 2.
            try: native._raw_request(f'env {native.env_id} close', _read_retry_count=0)
            except Exception: pass
            finally: native._resident_closed = True; native.close_transport()


if __name__ == '__main__':
    main()
