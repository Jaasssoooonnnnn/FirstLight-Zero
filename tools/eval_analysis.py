"""Tactical metadata from zero eval game records (sweeps/NAME/games/w*.jsonl): card combos and X-Bow / Tesla interplay.

    python tools/eval_analysis.py runs/RUN/sweeps/NAME [--by version|policy] [--out FILE]

Per side and game (side = 'small' or 'opp'; in a round robin both seats are zero and 'opp' is the other policy):
  espirit_log       E-Spirit and The Log by one owner within 1 s of each other (each play used once; same tick = one decision)
  tesla_preempt     own Tesla, then the enemy's X-Bow in the same lane within 5 s
  tesla_reactive    enemy X-Bow, then own Tesla in the same lane within 5 s
  xbow_fireballed   enemy X-Bow hit by own Fireball within 5 s (landing within 2.5 tiles of it)
  xbow_answered     share of enemy X-Bows met by own Tesla or Fireball within 5 s
  tesla_cover       own Tesla in the same lane as an own X-Bow placed within 5 s before or after it (protecting the X-Bow)
  xbow_lock_share   own X-Bows at row 12-16 of 32 (the spot that reaches the enemy princess tower from the own side)
  xbow_pocket_share own X-Bows on the enemy half (after taking a tower); the rest are deeper, defensive placements
  elixir_before_xbow, leaked elixir, game seconds
Lanes and depth use the owner's own frame (x, y in 0..1; y < .5 = own half), as side_profile in firstlight_zero/evaluate.py.
"""
import argparse, collections, glob, json
from pathlib import Path

XBOW, TESLA, FIREBALL, LOG, ESPIRIT = 27000008, 27000006, 28000000, 28000011, 26000084
WINDOW = 100          # 5 s at 20 ticks/s
PAIR = 20             # 1 s


def frame(owner, x, y):
    return (x / 18000., y / 32000.) if owner == 0 else (1 - x / 18000., 1 - y / 32000.)


def game_side(r, o):
    e = 1 - o
    mine = [ev for ev in r['events'] if ev[1] == o and ev[3] is not None]
    theirs = [ev for ev in r['events'] if ev[1] == e and ev[3] is not None]
    by = lambda evs, card: [ev for ev in evs if ev[2] == card]
    lane = lambda owner_frame, ev: frame(owner_frame, ev[3], ev[4])[0] < .5
    out = collections.Counter()
    # E-Spirit + Log pairs
    used = set()
    for a in by(mine, ESPIRIT):
        b = next((b for b in by(mine, LOG) if id(b) not in used and abs(b[0] - a[0]) <= PAIR), None)
        if b is not None:
            used.add(id(b)); out['espirit_log'] += 1; out['espirit_log_same_tick'] += a[0] == b[0]
    enemy_xbows = by(theirs, XBOW)
    out['enemy_xbow'] = len(enemy_xbows)
    for t in by(mine, TESLA):
        out['tesla'] += 1
        out['tesla_cover'] += any(abs(x[0] - t[0]) <= WINDOW and lane(o, t) == lane(o, x) for x in by(mine, XBOW))
        if any(t[0] < x[0] <= t[0] + WINDOW and lane(o, t) == lane(o, x) for x in enemy_xbows):
            out['tesla_preempt'] += 1
        if any(x[0] <= t[0] <= x[0] + WINDOW and lane(o, t) == lane(o, x) for x in enemy_xbows):
            out['tesla_reactive'] += 1
    for x in enemy_xbows:
        fb = [f for f in by(mine, FIREBALL) if x[0] <= f[0] <= x[0] + WINDOW and ((f[3] - x[3]) ** 2 + (f[4] - x[4]) ** 2) ** .5 <= 2500]
        ts = [t for t in by(mine, TESLA) if x[0] <= t[0] <= x[0] + WINDOW and lane(o, t) == lane(o, x)]
        out['xbow_fireballed'] += bool(fb); out['xbow_answered'] += bool(fb or ts)
    for x in by(mine, XBOW):
        row = int(frame(o, x[3], x[4])[1] * 32)
        out['xbow'] += 1; out['xbow_lock'] += 12 <= row < 17; out['xbow_pocket'] += row >= 17
        if x[6] is not None:
            out['elixir_before_xbow_sum'] += x[6]
    return out


def summarize(games, who):
    tot = collections.Counter(); n = len(games); leaked = 0.; secs = 0.
    for r in games:
        o = r['small_owner'] if who == 'small' else 1 - r['small_owner']
        tot.update(game_side(r, o)); leaked += r['elixir'][o]['leaked']; secs += r['tick'] / 20.
    per = lambda k: round(tot[k] / max(n, 1), 3)
    share = lambda a, b: round(tot[a] / tot[b], 3) if tot[b] else None
    return dict(games=n, espirit_log=per('espirit_log'), espirit_log_same_tick=per('espirit_log_same_tick'),
                tesla=per('tesla'), tesla_preempt=per('tesla_preempt'), tesla_reactive=per('tesla_reactive'), tesla_cover=per('tesla_cover'),
                enemy_xbow=per('enemy_xbow'), xbow_answered_share=share('xbow_answered', 'enemy_xbow'),
                xbow_fireballed_share=share('xbow_fireballed', 'enemy_xbow'),
                xbow=per('xbow'), xbow_lock_share=share('xbow_lock', 'xbow'), xbow_pocket_share=share('xbow_pocket', 'xbow'),
                elixir_before_xbow=round(tot['elixir_before_xbow_sum'] / tot['xbow'], 2) if tot['xbow'] else None,
                leaked=round(leaked / max(n, 1), 2), game_seconds=round(secs / max(n, 1), 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--by', default='version', choices=['version', 'policy'])
    ap.add_argument('--out')
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for f in sorted(glob.glob(str(Path(a.dir) / 'games' / 'w*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if not r['valid']:
                continue
            groups[r['version'] if a.by == 'version' else (r['version'], r['temperature'])].append(r)
            if a.by == 'policy' and r.get('opp_version') is not None:     # round robin: the other seat is a policy too
                groups[r['opp_version'], r['opp_temperature']].append(dict(r, small_owner=1 - r['small_owner']))
    rows = [dict(key=k, small=summarize(g, 'small'), opp=summarize(g, 'opp')) for k, g in sorted(groups.items())]
    text = json.dumps(rows, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
