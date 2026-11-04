"""Run status: throughput, per-frame timing, failures and the latest learner metrics."""
from pathlib import Path
import collections, glob, json, sys, time

run = Path(sys.argv[1]); now = time.time()
games, timing, failures = [], [], []
for f in glob.glob(str(run / 'collectors/*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        {'game': games, 'timing': timing, 'failure': failures}.get(r['event'], []).append(r)
for window in (300, 900, 3600):
    g = [x for x in games if x['utc'] > now - window]
    if not g:
        continue
    print(f'last {window:>4}s: {len(g) / window * 3600:8.0f} games/h  valid {sum(x["valid"] for x in g)}/{len(g)}'
          f'  mean_end_tick {sum(x["tick"] for x in g) / len(g):.0f}  rejected {sum(sum(x["rejected"]) for x in g)}'
          f'  cancelled {sum(sum(x.get("cancelled", [0, 0])) for x in g)}  acts/side {sum(sum(x["acts"]) for x in g) / len(g) / 2:.1f}')
recent = [x for x in timing if x['utc'] > now - 300]
if recent:
    total = collections.Counter()
    for x in recent:
        for k in ('features', 'policy', 'native', 'reward', 'reset', 'write', 'frames'):
            total[k] += x[k]
    print('per-frame ms', {k: round(1000 * v / total['frames'], 1) for k, v in total.items() if k != 'frames'},
          ' frames/min/collector', round(total['frames'] / len(recent)), ' collectors reporting', len({x.get('utc') and id(x) for x in recent}))
print('total games', len(games), ' failures', len(failures))
for f in sorted(glob.glob(str(run / 'logs/server-*.jsonl'))):   # inference servers: last minute
    r = json.loads(open(f).read().splitlines()[-1])
    if r['utc'] > now - 180:
        print(f"server {Path(f).stem}: {r['seats'] / 60:.0f} seats/s  {r['seats'] / max(r['batches'], 1):.0f} seats/batch"
              f"  gpu {1e3 * r['gpu_s'] / max(r['batches'], 1):.1f} ms/batch  busy {r['gpu_s'] / 60:.0%}  states {r['states']}")
for f in failures[-2:]:
    print('FAILURE', f['error'][-800:])
metrics = [json.loads(l) for l in open(run / 'metrics.jsonl')] if (run / 'metrics.jsonl').exists() else []
keys = ('version', 'episodes', 'stale', 'mean_len', 'act_rate', 'mean_reward', 'lag_mean', 'explained_var', 'update_s',
        'approx_kl', 'clip_frac', 'entropy', 'value_loss', 'vs_history')
for r in metrics[-3:]:
    print({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys if k in r})
for r in metrics[-1:]:
    if r.get('plays_per_game'):
        print(f"v{r['version']} plays/game {r['plays_per_game']}  game_s {r['game_seconds']:.0f}  acts {r['acts_per_game']:.1f}  crowns {r['crowns_per_game']:.2f}")
    if r.get('elixir_at_play'):
        print(f"v{r['version']} elixir at play {r['elixir_at_play']}  frames at 10 elixir {r['frames_at_10']:.1%}  gamma {r.get('gamma')}")
pfsp = run / 'weights/pfsp.json'
if pfsp.exists():
    stats = {int(v): x for v, x in json.loads(pfsp.read_text())['stats'].items()}
    p = {v: (n * sc + 1) / (n + 2) for v, (n, sc) in stats.items()}
    w = {v: (1 - x) ** 2 for v, x in p.items()}; total = sum(w.values()) or 1
    top = sorted(w, key=w.get, reverse=True)[:6]
    print('pfsp hardest', [(v, round(p[v], 2), f'{w[v] / total:.0%}') for v in top], ' snapshots', len(stats))
evals = run / 'evals.jsonl'
if evals.exists():
    for line in evals.read_text().splitlines()[-6:]:
        e = json.loads(line); m = e.get('margin') or {}
        print(f"vs IL v{e['version']} {e['small_mode']}: {e['wins']}-{e['draws']}-{e['losses']} score {e['score_rate']:.0%}"
              + (f"  crowns {m['small_crowns']:.2f}:{m['il_crowns']:.2f}  tower_hp {m['small_tower_hp']:.2f}:{m['il_tower_hp']:.2f}" if m else ''))
events = run / 'node-events.jsonl'
print('collector restarts', sum(1 for _ in open(events)) if events.exists() else 0,
      ' supervisor events', (run / 'supervisor-events.jsonl').read_text() if (run / 'supervisor-events.jsonl').exists() else 'none',
      ' inbox', len(list((run / 'inbox').glob('*.npz'))))
