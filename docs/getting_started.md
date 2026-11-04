# Getting started

## 1. Engine and game data

FirstLight Zero drives native battle engines over the FirstLight resident protocol. Nothing in this repository can produce
them on its own: you bring the game client, and you are responsible for your right to use it.

| Requirement | Value |
|---|---|
| Game version | 15.535.13 |
| Content version | 15.535.86 |
| Probe source | `native_runner/probe/`, the version this code expects |
| Resident slots | 8 per engine |

[FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) documents how to build an engine from a user-supplied client
([training guide](https://github.com/Jaasssoooonnnnn/FirstLight_CR/blob/main/docs/training.md)).
[daybreak-cr](https://github.com/Jason-XII/daybreak-cr) describes the extracted engine approach. The published results were
produced on Linux with an extracted engine; emulator-hosted engines were not tested with this code.

`CR_WORKSPACE_ROOT` must point to a directory holding the game data of that version:

```text
$CR_WORKSPACE_ROOT/
  runtime-update/                    fingerprint.json, csv_logic/, assets.scdb, ...
  nr_15.535.13_release_<id>/         decoded release
  captures/
    decompressed_assets/
```

## 2. Engine list

Copy `engines.example.json` to `engines.json`, one entry per engine process. Or point `FIRSTLIGHT_ZERO_ENGINES` at the file.

```json
{"collectors": [{"host": "localhost", "port": 26789, "env_ids": [0, 1, 2, 3, 4, 5, 6, 7], "cpu": 2}]}
```

Engines listen on loopback, so each worker runs on its engine's host. Remote hosts are started with `ssh`, and the repository
and run directory must be on a path shared between machines. `cpu` is optional and pins the worker on Linux.

## 3. Train

```bash
python firstlight_zero/launch.py prepare --run runs/first
python firstlight_zero/launch.py start --run runs/first > runs/first/supervisor.log 2>&1 &
python tools/status.py runs/first
```

`prepare` writes `settings.json`. Override any default in `firstlight_zero/launch.py` with `--set`, for example to spread the
learner and servers over several GPUs:

```bash
python firstlight_zero/launch.py prepare --run runs/first --set '{"learner_gpus": [0, 1], "server_gpus": [2, 3], "servers": 4}'
```

`touch runs/first/STOP` stops the run after the current work. `start` resumes from `learner-state.pt`.

Run outputs:

| Path | Contents |
|---|---|
| `metrics.jsonl` | One line per version: losses, entropy, KL, explained variance, plays per card, elixir |
| `weights/history/v{N}.pt` | Snapshots used as opponents and for evaluation |
| `weights/pfsp.json` | Recent score of the current policy against each snapshot |
| `collectors/*.jsonl` | Per-game results and timing |
| `logs/` | Process logs and per-minute server statistics |

## 4. Use the pretrained model

Evaluation reads snapshots from a run directory, so give the checkpoint one:

```bash
python firstlight_zero/launch.py prepare --run runs/pretrained
mkdir -p runs/pretrained/weights/history
cp checkpoints/firstlight-zero-v1450.pt runs/pretrained/weights/history/v1450.pt
python firstlight_zero/evaluate.py sweep --run runs/pretrained --name check --versions 1450 --temps 0.5 --games 40 \
    --opponent /path/to/v4-checkpoint.pt
```

The other versions in `checkpoints/` work the same way. The opponent is any V4 checkpoint from the FirstLight model family. See [evaluation and replays](evaluation.md).

## 5. Tests

`tests/test_consistency.py` needs no engine. It checks that step-by-step acting and whole-episode training give the same
probabilities and values after a round trip through the trajectory store.

```bash
python tests/test_consistency.py
```
