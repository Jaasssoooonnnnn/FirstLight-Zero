<p align="center">
  <img src="docs/assets/firstlight-zero.png" width="240" alt="FirstLight Zero logo">
</p>
<h1 align="center">FirstLight Zero</h1>
<p align="center">A small Clash Royale model that learned the 3.0 X-Bow mirror from self-play alone, with no human data.</p>
<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Learning-PPO_self--play-EE4C2C.svg" alt="PPO self-play">
  <img src="https://img.shields.io/badge/Parameters-1.17M-1565c0.svg" alt="1.17M parameters">
</p>
<p align="center">
  <a href="docs/training_history.md">Training history</a> ·
  <a href="docs/getting_started.md">Getting started</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/evaluation.md">Evaluation</a> ·
  <a href="checkpoints/README.md">Model</a>
</p>

Every strong Clash Royale AI I know of starts from human replays. I wanted to see whether a model could get there without them, so this one started from random weights and only ever played against itself and its older versions. Nobody told it what the deck does.

After 1,450 training rounds, the 1.17M-parameter model beats a 16.8M-parameter imitation model that was trained on 193,530 top ladder games.

<p align="center">
  <img src="docs/assets/vs-boss.png" width="800" alt="Win rate against the imitation model rises from 0% before v270 to above 90% by v1300">
</p>

## Results

40 games, in the X-Bow mirror at level 11. `checkpoints/` holds every 50th version from v50 to v1500; v1450 is the one to use.

| Opponent | Size | Trained on | Our version | Win rate |
|---|---|---|---|---|
| IL Model Everlight | 16.8M | 193,530 human games | v1460 | 97.5% |

It was never part of training. The [training history](docs/training_history.md) has the whole curve and how the model's play changed version by version.

## What's here

| Path | Contents |
|---|---|
| `firstlight_zero/` | Training and evaluation: features, model, collectors, inference servers, PPO learner, sweeps and round robins |
| `native_runner/` | The battle environment layer, trimmed from FirstLight CR, with the probe source in `native_runner/probe/` |
| `tools/` | Evaluation servers, tactics analysis, replay export and replay checks |
| `checkpoints/` | Every 50th version, `firstlight-zero-v50.pt` to `firstlight-zero-v1500.pt` |
| `docs/` | The guides linked above |

## Quick start

You need Python 3.12, a GPU for training, and native battle engines for version 15.535.13. This repository doesn't include the game client. [FirstLight CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) has the full stack for building an engine from a client you supply: probe, APK tools, environment and a training guide. [Getting started](docs/getting_started.md) lists exactly what this code expects.

```bash
pip install -r requirements.txt
python tests/test_consistency.py      # needs no engine

cp engines.example.json engines.json  # one entry per engine process
python firstlight_zero/launch.py prepare --run runs/first
python firstlight_zero/launch.py start --run runs/first > runs/first/supervisor.log 2>&1 &
python tools/status.py runs/first
```

To train with different settings, pass `--set` to `prepare`. To evaluate the included model, see [Use the pretrained model](docs/getting_started.md#4-use-the-pretrained-model).

## Acknowledgements

Thanks to [daybreak-cr](https://github.com/Jason-XII/daybreak-cr). Its asynchronous training framework inspired the one here, which made my PPO training throughput 1.5 times faster than before.

## License

FirstLight Zero is an independent offline project under [Apache-2.0](LICENSE), unaffiliated with Supercell. It doesn't distribute the game or its resources and doesn't connect to official servers. See [NOTICE](NOTICE).
