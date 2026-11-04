"""Exports zero eval / sweep games as a replay dataset in the IL raw-data layout (RoyaleAPI collector payloads in parquet).

Each game becomes one row with the columns of `royaleapi-replay-parquet.v1` (+ eval columns) and a `payload_json` that
`native_runner.royaleapi_replay.prepare_collected_replay` accepts (schema royaleapi-battle-actions.v1: battle header with
decks/levels/towers/crowns, play_card / activate_ability events in the collector's team-at-bottom frame with data_i).
The payload also carries an `zero` block with everything the RoyaleAPI format cannot hold:
- native: the exact episode (seed, deck order, rules) and every submitted action with its decision tick and fate,
  which reproduces the game tick for tick (`tools/replay_check.py`)
- eval metadata (snapshot, temperature, opponent, protocol), result, elixir/tower trace, building/hero board trace.

    python tools/replay_export.py DIR [--check 20]       DIR = an eval dir (evals/v1500-sample) or a sweep dir (sweeps/NAME)
writes DIR/replays/part-000000.parquet ... and DIR/replays/manifest.json
"""
import argparse, collections, datetime, hashlib, json, random, sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

KEYS = {27000008: ('x-bow', 'X-Bow', 'building', 6), 26000000: ('knight-hero', 'Knight', 'troop', 3),
        26000001: ('archers-ev1', 'Archers', 'troop', 3), 27000006: ('tesla-ev1', 'Tesla', 'building', 4),
        26000010: ('skeletons', 'Skeletons', 'troop', 1), 26000084: ('electro-spirit', 'Electro Spirit', 'troop', 1),
        28000000: ('fireball', 'Fireball', 'spell', 4), 28000011: ('the-log', 'The Log', 'spell', 2)}
BASE = {'knight-hero': 'knight', 'archers-ev1': 'archers', 'tesla-ev1': 'tesla'}
LEVEL = 11
TAG_ALPHABET = '0289PYLQGRJCUV'     # the loader accepts only Clash Royale tag characters


def tower_hitpoints(towers, owner):
    """RoyaleAPI final_tower_hitpoints for one owner; left/right from that owner's own view (owner 1 is rotated)."""
    out = dict(king=0, princess_left=0, princess_right=0)
    princess = sorted([t for t in towers if 'king' not in t[0]], key=lambda t: t[1] if owner == 0 else -t[1])
    for t in towers:
        if 'king' in t[0]: out['king'] = t[3]
    for name, t in zip(('princess_left', 'princess_right'), princess):
        out[name] = t[3]
    out['total'] = out['king'] + out['princess_left'] + out['princess_right']
    return out


def coordinates(x, y, data_i):
    """Native arena units -> the collector's team-at-bottom frame (inverse of royaleapi_replay._oriented_grid_target)."""
    px, py = (x, 32000 - y) if data_i == 0 else (18000 - x, y)
    gx, gy = min(max(int(px // 1000), 0), 17), min(max(int(py // 1000), 0), 31)
    ox, oy = round(px / 1000. - (gx + .5), 4), round(py / 1000. - (gy + .5), 4)
    dx, dy = px, 32000 - py
    raw = (dx, dy) if data_i == 0 else (18000 - dx, 32000 - dy)
    return dict(arena_label_from_blue=dict(column=chr(ord('A') + gx), row=gy + 1), display_royaleapi_units=dict(x=dx, y=dy),
                grid_cell_floor=dict(x=gx, y=gy), inside_18x32_arena=True, inversion_applied=data_i == 1,
                native_world_units=dict(x=px, y=py), raw_royaleapi_units=dict(x=raw[0], y=raw[1]),
                subcell_offset_from_floor_center=dict(x=ox, y=oy)), raw


def player_block(owner, rec, name, tag, opponent_owner):
    deck = [dict(card_key=KEYS[c][0], level=LEVEL, name=KEYS[c][1]) for c in rec['native']['layouts'][owner]['card_ids']]
    return dict(average_elixir=3.0, deck=deck, elixir_leaked=rec['elixir'][owner].get('leaked'),
                final_tower_hitpoints=tower_hitpoints(rec.get('towers_final', [[], []])[owner], owner), four_card_cycle_elixir=7.0,
                name=name, tag=tag, tower_card=dict(card_key='tower-princess', level=LEVEL, name='Tower Princess'), trophy_values=[0, 0])


def execution_order(rec):
    """Play events with same-tick plays of one owner (a multi-card decision, e.g. Log + E-Spirit) in the order the actions were
    issued; observation events list them by card id, and the loader's card-cycle inference needs the real order."""
    plays = [e for e in rec['events'] if e[3] is not None]
    if 'native' not in rec:
        return plays
    issued = {o: [a['card_id'] for _, a, _ in rec['native']['actions'] if a['owner'] == o and a.get('kind') == 'play_card'] for o in (0, 1)}
    cursor = {0: 0, 1: 0}; rank = {}
    groups = collections.defaultdict(list)
    for i, e in enumerate(plays):
        groups[e[1], e[0]].append(i)
    for (owner, _), members in sorted(groups.items(), key=lambda item: item[0][1]):
        seq = issued[owner]; found = []
        for i in members:
            j = next((k for k in range(cursor[owner], len(seq)) if seq[k] == plays[i][2] and k not in found), len(seq) + i)
            found.append(j); rank[i] = j
        cursor[owner] = max(cursor[owner], min(max(found) + 1, len(seq)))
    return [plays[i] for i in sorted(range(len(plays)), key=lambda i: (plays[i][0], rank[i]))]


def payload(rec, meta, replay_tag, delay):
    s = rec['small_owner']; o = 1 - s
    data_i = 0 if s == 1 else 1          # team = the small policy; owner 1 is the bottom (True Blue) seat
    side = {s: 'team', o: 'opponent'}
    v, t = rec.get('version'), rec.get('temperature')
    small_name = f"zero v{v}" + (f" T{t:g}" if t is not None else '')
    opp_name, opp_tag = meta.get('opponent', 'opponent'), 'OPPONENT'
    if rec.get('opp_version') is not None:     # round robin: the opponent is another zero policy
        ot = rec.get('opp_temperature')
        opp_name, opp_tag = f"zero v{rec['opp_version']}" + (f" T{ot:g}" if ot is not None else ''), f"ZEROV{rec['opp_version']}"
        if opp_tag == f'ZEROV{v}':
            opp_tag += 'B'
    events = []; numbers = collections.Counter(); counts = {'team': collections.Counter(), 'opponent': collections.Counter()}
    agg = {k: collections.defaultdict(lambda: [0, 0]) for k in ('team', 'opponent')}
    timeline = [('play', e) for e in execution_order(rec)] + [('ability', (tick, own)) for own in (0, 1) for tick in rec['ability'][own]]
    timeline.sort(key=lambda item: item[1][0])
    for index, (kind, e) in enumerate(timeline):
        if kind == 'play':
            tick, owner, card, x, y, cost, e_own, e_opp = e
            key, name, category, base_cost = KEYS[card]; sd = side[owner]; numbers[sd, key] += 1
            coords, raw = coordinates(x, y, data_i)
            counts[sd][BASE.get(key, key)] += 1; agg[sd][category][0] += 1; agg[sd][category][1] += int(cost if cost is not None else base_cost)
            events.append(dict(ability_source_authoritative=None, ability_source_candidates=[], card_key=key, card_play_number=numbers[sd, key],
                               coordinates=coords, deck_card_key_candidates=[key], form_at_play='unknown', kind='play_card',
                               replay_tick_20hz=tick - delay, side=sd,
                               source_fields=dict(data_c=key, data_i=data_i, data_s=sd[0], data_t=tick - delay, data_x=raw[0], data_y=raw[1],
                                                  timeline_card_key=key),
                               source_index=index, time_seconds=round((tick - delay) / 20., 2),
                               zero=dict(native_tick=tick, owner=owner, native_x=x, native_y=y, cost=cost, elixir_before=e_own, enemy_elixir_before=e_opp)))
        else:
            tick, owner = e; sd = side[owner]; numbers[sd, 'ability'] += 1; agg[sd]['ability'][0] += 1
            events.append(dict(ability_source_authoritative=False, ability_source_candidates=['knight-hero'], card_key=None,
                               card_play_number=numbers[sd, 'ability'], coordinates=None, deck_card_key_candidates=[], form_at_play=None,
                               kind='activate_ability', replay_tick_20hz=tick - delay, side=sd,
                               source_fields=dict(data_c='_invalid', data_i=data_i, data_s=sd[0], data_t=tick - delay, data_x=None, data_y=None,
                                                  timeline_card_key='_invalid'),
                               source_index=index, time_seconds=round((tick - delay) / 20., 2), zero=dict(native_tick=tick, owner=owner)))
    result = 'victory' if rec['small_result'] > 0 else 'defeat' if rec['small_result'] < 0 else 'draw'
    for sd in agg:
        agg[sd]['total'] = [sum(agg[sd][k][0] for k in ('troop', 'spell', 'building')), sum(agg[sd][k][1] for k in ('troop', 'spell', 'building'))]
    seconds = rec['tick'] / 20.
    battle = dict(battle_type='firstlight-zero-eval', game_mode='Ranked', played_at_local=None, played_at_utc=None, requested_player_tag=f'ZEROV{v}',
                  result=result,
                  team=dict(crowns=rec['crowns'][s], players=[player_block(s, rec, small_name, f'ZEROV{v}', o)], tags=[f'ZEROV{v}']),
                  opponent=dict(crowns=rec['crowns'][o], players=[player_block(o, rec, opp_name, opp_tag, s)], tags=[opp_tag]))
    keep = ('game', 'seed', 'small_owner', 'version', 'temperature', 'opp_version', 'opp_temperature', 'opp_model', 'result', 'small_result', 'crowns', 'tower_hp', 'towers_final', 'tick',
            'acts', 'first_act_tick', 'forced_tick', 'rejected', 'cancelled', 'valid', 'truncated', 'elixir', 'ability', 'events', 'trace',
            'board', 'native', 'plays')
    return dict(battle=battle, collection_schema_version='firstlight-zero-eval-replay-collection.v1', collection_worker=dict(account_id='firstlight-zero-eval'),
                events=events,
                replay=dict(aggregate_stats={sd: dict({k: list(agg[sd][k]) for k in ('ability', 'building', 'spell', 'troop', 'total')},
                                                      leaked=rec['elixir'][s if sd == 'team' else o].get('leaked')) for sd in agg},
                            card_counts={sd: dict(counts[sd]) for sd in counts},
                            duration=dict(display_label=f'{int(seconds // 60)}:{int(seconds % 60):02d}', display_seconds=int(seconds),
                                          timeline_seconds=seconds)),
                schema_version='royaleapi-battle-actions.v1',
                source=dict(replay_tag=replay_tag, site='firstlight-zero-eval', run=Path(meta.get('run', '')).name, eval=meta.get('name') or meta.get('version')),
                validation=dict(ability_event_count=sum(e['kind'] == 'activate_ability' for e in events),
                                coordinate_event_count=sum(e['kind'] == 'play_card' for e in events),
                                normalized_event_count=len(events), warnings=[]),
                zero=dict(schema='firstlight-zero-eval-game.v1', protocol={k: meta.get(k) for k in (
                    'opponent', 'il_sha256', 'il_mode', 'delay_policy', 'allowed_delay_offsets_ms', 'delay_ceiling_ms',
                    'forced_opening', 'force_after_seconds', 'execution_jitter_ticks', 'battle_level', 'deck', 'small_mode')},
                         small_checkpoint=Path(rec['model']).name if rec.get('model') else None, **{k: rec.get(k) for k in keep if k != 'model'}))


def export(directory, part_rows=500, check=0):
    import pyarrow as pa, pyarrow.parquet as pq
    from native_runner.royaleapi_replay import ROYALAPI_COMMAND_TO_NATIVE_OBSERVABLE_TICKS as delay
    directory = Path(directory).resolve(); meta = json.loads((directory / 'config.json').read_text())
    records = [json.loads(l) for f in sorted((directory / 'games').glob('w*.jsonl')) for l in f.read_text().splitlines()]
    records.sort(key=lambda r: r['game'])
    out = directory / 'replays'; out.mkdir(exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    rows = []; manifest = []
    for rec in records:
        if 'native' not in rec:
            continue
        h = int(hashlib.sha1(f"{meta.get('run')}|{directory.name}|{rec['game']}".encode()).hexdigest(), 16)
        tag = ''.join(TAG_ALPHABET[h // len(TAG_ALPHABET) ** k % len(TAG_ALPHABET)] for k in range(14))
        p = payload(rec, meta, tag, delay)
        b = p['battle']
        rows.append(dict(schema_version='royaleapi-replay-parquet.v1', replay_tag=tag, requested_player_tag=b['requested_player_tag'],
                         played_at_utc=datetime.datetime.fromtimestamp(rec.get('utc', now.timestamp()), datetime.timezone.utc),
                         played_at_local=None, fetched_at=now, battle_type=b['battle_type'], game_mode=b['game_mode'], result=b['result'],
                         team_tags=b['team']['tags'], opponent_tags=b['opponent']['tags'], team_crowns=b['team']['crowns'],
                         opponent_crowns=b['opponent']['crowns'], account_id='firstlight-zero-eval', proxy_id='', event_count=len(p['events']),
                         warning_count=0, warnings=[], source_page_url='', payload_json=json.dumps(p, separators=(',', ':')),
                         version=rec.get('version'), temperature=rec.get('temperature'), opponent=meta.get('opponent'),
                         opp_version=rec.get('opp_version'), opp_temperature=rec.get('opp_temperature'),
                         small_owner=rec['small_owner'], seed=rec['seed'], small_result=rec['small_result'], valid=rec['valid'],
                         end_tick=rec['tick'], eval_game=rec['game']))
    for k in range(0, len(rows), part_rows):
        name = f'part-{k // part_rows:06d}.parquet'
        pq.write_table(pa.Table.from_pylist(rows[k:k + part_rows]), out / name, compression='zstd')
        manifest += [dict(replay_tag=r['replay_tag'], source_file=f'replays/{name}', source_row=j) for j, r in enumerate(rows[k:k + part_rows])]
    (out / 'manifest.json').write_text(json.dumps(dict(count=len(manifest), source=str(directory), rows=manifest), indent=0))
    report = dict(games=len(rows), parts=(len(rows) + part_rows - 1) // part_rows)
    if check:
        report['check'] = roundtrip([json.loads(r['payload_json']) for r in random.Random(0).sample(rows, min(check, len(rows)))])
    return report


def roundtrip(payloads):
    """prepare_collected_replay on our payloads: every play must come back with the native owner, card, point (grid + subcell offset) and
    observable tick of the action the engine executed."""
    from native_runner.royaleapi_replay import prepare_collected_replay
    from native_runner.contracts import ActionKind
    bad = []; plays = 0; abilities = 0
    for p in payloads:
        try:
            prepared = prepare_collected_replay(p)
        except Exception as error:
            bad.append(dict(tag=p['source']['replay_tag'], error=repr(error)[:300])); continue
        got = []
        for op in prepared.operations if hasattr(prepared, 'operations') else prepared.replay.operations:
            for a in op.actions:
                if a.kind is ActionKind.PLAY_CARD:
                    off = a.subcell_offset or (0., 0.)     # a point on a cell line may come back as (cell, +.5) or (cell + 1, -.5): compare points
                    got.append((op.end_native_tick, a.owner, a.card_id, tuple(round(g + .5 + o, 3) for g, o in zip(a.target_grid, off))))
                elif a.kind is ActionKind.ACTIVATE_ABILITY:
                    abilities += 1
        want = []
        for e in p['events']:
            if e['kind'] != 'play_card':
                continue
            x, y = e['zero']['native_x'], e['zero']['native_y']
            want.append((e['zero']['native_tick'], e['zero']['owner'], [c for c, k in KEYS.items() if k[0] == e['card_key']][0], (round(x / 1000., 3), round(y / 1000., 3))))
        plays += len(want)
        if sorted(got) != sorted(want):
            diff = [w for w in sorted(want) if w not in got][:3]
            bad.append(dict(tag=p['source']['replay_tag'], plays=len(want), returned=len(got), first_missing=diff, first_got=sorted(got)[:3]))
    return dict(payloads=len(payloads), plays=plays, abilities=abilities, mismatched=len(bad), examples=bad[:5])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--check', type=int, default=0)
    ap.add_argument('--part-rows', type=int, default=500)
    a = ap.parse_args()
    print(json.dumps(export(a.dir, a.part_rows, a.check), indent=1))
