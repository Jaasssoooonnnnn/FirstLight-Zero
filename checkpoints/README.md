# Model

`firstlight-zero-v1450.pt` is version 1450 of the run in the [training history](../docs/training_history.md), selected by round
robin among 50 snapshot and temperature combinations.

| | |
|---|---|
| Architecture | `XbowNet`: channels 64, dilations 1/2/4/8, scatter 16, hidden 256 |
| Parameters | 1.17M |
| Format | PyTorch `state_dict` |
| Sampling temperature | 0.5 on gate, card and location |
| Deck | 3.0 X-Bow mirror only, level 11 |
| Game version | 15.535.13 |

The temperature is part of the result. Training sampled at 1.0, which plays clearly worse: about 66 Elo below 0.5 on average.

The opponents used in the published results are not included.
