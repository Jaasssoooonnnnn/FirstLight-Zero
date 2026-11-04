"""Exact-replay check of zero eval game records: rebuild the episode from the recorded matchup (seed, deck, level), feed the
recorded actions at their decision ticks to an engine, and compare the outcome with the record (final tick,
crowns, tower hp, every executed play with tick/owner/card/position). No policy runs; one engine.

    python tools/replay_check.py DIR [--games 8]          DIR = an eval dir or a sweep dir with games/w*.jsonl
writes DIR/replay-check/result.json
"""
import argparse, collections, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from firstlight_zero.hosts import load_engines, pin, spawn_on   # noqa: E402


def node(inp, out):
    from native_runner.cr_native_env import ResidentNativeClashEnv
    from native_runner.resident_batch_vector import ResidentBatchCoordinatorV1, ResidentBatchNativeProxyV1
    from native_runner.training.v4.ppo_runtime import ppo_environment_config_v4, build_ppo_environment_v4, episode_for_matchup_v4
    from native_runner.training.v4.deck_pool import DeckPoolManifestV4
    from native_runner.training.v4.matchmaking import ScheduledMatchupV4, POLICY_CURRENT
    from native_runner.contracts import ActionV1
    from firstlight_zero.spec import XBOW_DECK_ID, POOL, GAMEPLAY_END_TICK
    from firstlight_zero.actions import wait_action
    data = json.loads(Path(inp).read_text()); spec = data['engine']
    pin([spec.get('cpu')])
    native = ResidentNativeClashEnv(spec['env_ids'][0], '127.0.0.1', spec['port'], timeout=120.)
    coord = ResidentBatchCoordinatorV1((native,), timeout=120., training_capture=True, compressed_json=False, validate_json_capture=False)
    cfg = ppo_environment_config_v4(gamma_per_decision=.9997, shaping_beta=0.)
    env = build_ppo_environment_v4(ResidentBatchNativeProxyV1(native, coord), cfg)
    env.simulated_card_delay_jitter_ticks = 0; env.prepare_runner_attestation()
    deck = next(d for d in DeckPoolManifestV4.load(POOL).decks if d.deck_id == XBOW_DECK_ID)
    results = []
    for rec in data['records']:
        started = time.time(); nm = rec['native']; mu = nm['matchup']
        m = ScheduledMatchupV4(sequence=mu['sequence'], category=mu['category'], policy_matchup=POLICY_CURRENT, deck0=deck, deck1=deck,
                               current_owners=(0, 1), battle_level=mu['battle_level'], seed=mu['seed'])
        episode = episode_for_matchup_v4(m, cfg)
        same_episode = json.loads(json.dumps(episode.to_dict())) == nm['episode']
        obs, _ = env.reset(episode, options=nm['reset_options']); coord.start_session()
        plan = collections.defaultdict(lambda: collections.defaultdict(list))
        for t, a, _fate in nm['actions']:
            plan[int(t)][int(a['owner'])].append(ActionV1.from_mapping(a))
        seen = set(); events = []

        def collect(o_):
            for o in (0, 1):
                for e in o_[o].events:
                    if e.event_type != 'action_executed' or e.owner is None or e.card_id is None:   # hero ability: compared via ability_events
                        continue
                    pos = tuple(int(round(float(v))) for v in (e.position or ()))
                    key = (int(e.tick), int(e.owner), int(e.card_id), pos)
                    if key not in seen:
                        seen.add(key); events.append([int(e.tick), int(e.owner), int(e.card_id), pos[0] if pos else None, pos[1] if len(pos) > 1 else None])
        while True:
            collect(obs); tick = int(obs[0].tick)
            cmds = {o: (tuple(plan[tick][o]) if tick < GAMEPLAY_END_TICK and plan[tick][o] else (wait_action(o),)) for o in (0, 1)}
            coord.begin_round((native.env_id,))
            try:
                nxt, _, ended, truncated, _ = env.step(cmds, record_trace=False, policy_features_only=True)
            except BaseException as error:
                coord.abort_round(error); raise
            coord.finish_round(); obs = nxt
            if ended['__all__'] or truncated['__all__']:
                collect(obs); break
        coord.close_session()
        final = obs[0]
        crowns = [int(next(q for q in final.players if q.owner == o).crowns) for o in (0, 1)]
        tower_hp = [round(sum(float(t.hitpoints or 0) for t in final.towers if t.owner == o)
                          / max(sum(float(t.max_hitpoints or 0) for t in final.towers if t.owner == o), 1.), 4) for o in (0, 1)]
        want = sorted(e[:5] for e in rec['events'])
        got = sorted(events)
        results.append(dict(game=rec['game'], same_episode=same_episode, tick=[rec['tick'], int(final.tick)], crowns=[rec['crowns'], crowns],
                            tower_hp=[rec['tower_hp'], tower_hp], plays=[len(want), len(got)],
                            identical=same_episode and rec['tick'] == int(final.tick) and rec['crowns'] == crowns and rec['tower_hp'] == tower_hp and want == got,
                            first_diff=next(([w, g] for w, g in zip(want, got) if w != g), None), seconds=round(time.time() - started, 1)))
        Path(out).write_text(json.dumps(results, indent=1))
    native.close_transport()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('action_or_dir'); ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--input'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.action_or_dir == 'node':
        return node(a.input, a.out)
    directory = Path(a.action_or_dir).resolve(); work = directory / 'replay-check'; work.mkdir(exist_ok=True)
    recs = [json.loads(l) for f in sorted((directory / 'games').glob('w*.jsonl')) for l in f.read_text().splitlines()]
    recs = [r for r in recs if 'native' in r]
    recs = random.Random(0).sample(recs, min(a.games, len(recs)))
    spec = dict(load_engines()['collectors'][0], engine=0)
    (work / 'input.json').write_text(json.dumps(dict(engine=spec, records=recs)))
    step = spawn_on(spec['host'], [sys.executable, str(Path(__file__).resolve()), 'node', '--input', str(work / 'input.json'),
                                   '--out', str(work / 'result.json')], None)
    if step.wait():
        raise SystemExit('replay check failed')
    results = json.loads((work / 'result.json').read_text())
    print(json.dumps(dict(games=len(results), identical=sum(r['identical'] for r in results), results=results), indent=1))


if __name__ == '__main__':
    main()
