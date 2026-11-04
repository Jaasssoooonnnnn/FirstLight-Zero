# Evaluation and replays

## Protocol

Recorded in every result (`config.json`, `il-protocol.json`).

- Opponent on argmax. Zero policy sampled at a temperature that divides the gate, card and location logits; 0 means argmax.
- Earliest legal delay on both sides, ceiling 200 ms (bins 0, 50, 100, 150, 200). The opponent's delay mask is set after loading
  and checked.
- Forced opening for both sides after 15 seconds; the policy picks card and tile.
- No execution jitter.

## Servers

CPU inference works but is slow for large sweeps. GPU servers use the same code as training:

```bash
python tools/eval_servers.py start runs/NAME --opponent OPP.pt --zero-gpus 0 --opponent-gpus 0,0 > runs/NAME/eval-servers.log 2>&1 &
python tools/eval_servers.py stop runs/NAME
```

Add `--inference server --il-inference server` to the commands below. Repeat a GPU id to run several servers on it: an opponent
server's CPU work is limited by the GIL, so large engine counts want more than one.

## Sweep and round robin

```bash
# every 10th snapshot x 40 games, same seeds for all snapshots
python firstlight_zero/evaluate.py sweep --run runs/NAME --name sweep --versions 10:1500:10 --games 40 --schedule single --opponent OPP.pt

# best 10 snapshots x 5 temperatures, every pair once from each seat
python firstlight_zero/evaluate.py roundrobin --run runs/NAME --name final --top-from sweep --top 10 --temps 0,0.5,0.7,1,1.3 --seeds-per-pair 1
```

Results land in `runs/NAME/sweeps/NAME/`: `summary.csv` and `result.json` for a sweep; `ranking`, `mean_elo_by_version` and
`mean_elo_by_temperature` in `result.json` for a round robin. Bradley–Terry uses a weak L2 prior, with bootstrap intervals and
P(best).

Every game record in `games/w*.jsonl` holds both sides' plays with exact elixir before each play, Hero ability ticks, an elixir
and tower trace, and X-Bow, Tesla and Hero Knight positions and health once per second.

## Replays

```bash
python tools/replay_export.py runs/NAME/sweeps/NAME --check 100
python tools/replay_check.py runs/NAME/sweeps/NAME --games 8
```

`replay_export.py` writes `replays/part-*.parquet`, one row per game. `payload_json` is a `royaleapi-battle-actions.v1` payload that
`native_runner.royaleapi_replay.prepare_collected_replay` reads, the same format as the FirstLight replay data. Its `zero` block
keeps what that format cannot hold: the exact episode and every submitted action. `--check` verifies that owner, card, position and
tick survive the round trip through the loader.

`replay_check.py` rebuilds the episode in an engine, feeds the recorded actions at their decision ticks, and compares final tick,
crowns, tower health and every executed play.

## Tactics

```bash
python tools/eval_analysis.py runs/NAME/sweeps/NAME --by version --out tactics.json
```

Counts combos and X-Bow and Tesla interplay per side. The definitions are at the top of the script.
