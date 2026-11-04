"""act() step by step vs evaluate() on whole episodes, round-tripped through SeatStore -> .npz -> collate."""
from pathlib import Path
import json, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
import numpy as np
import torch

from firstlight_zero.model import XbowNet
from firstlight_zero.store import SeatStore, load
from firstlight_zero.learner import collate, gae
from firstlight_zero.spec import (CARD_FEATURES, HEIGHT, MAX_UNITS, NO_CARD, OPP_CARD_FEATURES, OPTIONS, OPTION_FEATURES, SCALARS,
                      UNIT_FEATURES, UNIT_VOCAB, WIDTH)


def random_step(rng):
    n = int(rng.integers(6, 30))
    unit_type = np.zeros(MAX_UNITS, np.int64); unit_type[:n] = rng.integers(1, UNIT_VOCAB, n)
    unit_feat = np.zeros((MAX_UNITS, UNIT_FEATURES), np.float32); unit_feat[:n] = rng.random((n, UNIT_FEATURES))
    unit_feat[:n, 0] = rng.integers(0, 2, n)
    inputs = dict(scal=rng.random(SCALARS).astype(np.float32), cards=rng.random((8, CARD_FEATURES)).astype(np.float32),
                  opp_cards=rng.random((8, OPP_CARD_FEATURES)).astype(np.float32), option_card=rng.integers(0, NO_CARD + 1, OPTIONS),
                  opt_feat=rng.random((OPTIONS, OPTION_FEATURES)).astype(np.float32), unit_type=unit_type, unit_feat=unit_feat, unit_n=np.int64(n))
    opt_mask = rng.random(OPTIONS) < .5
    place = rng.random((4, HEIGHT, WIDTH)) < .3; place[:, 0, 0] = True
    return inputs, dict(opt_mask=opt_mask, place=place, opt_bin=np.zeros(OPTIONS, np.int64), force=bool(rng.random() < .05))


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XbowNet().to(device).eval()
    print('parameters', sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        model.gate.bias.copy_(torch.tensor([0., 0.]))
    tmp = Path(tempfile.mkdtemp()); episodes = []
    for ep in range(4):
        store = SeatStore(); state = model.initial_state(1, device); length = int(rng.integers(40, 90))
        for t in range(length):
            inputs, legal = random_step(rng)
            x = {k: torch.from_numpy(np.asarray(v)[None]).to(device) for k, v in inputs.items() if k != 'unit_n'}
            lg = dict(opt_mask=torch.from_numpy(legal['opt_mask'][None]).to(device), place=torch.from_numpy(legal['place'][None]).to(device),
                      force=torch.tensor([legal['force']], device=device))
            out, state = model.act(x, lg, state)
            d = {k: v[0].item() for k, v in out.items()}
            store.append(inputs, legal, d, 0); store.reward(float(rng.normal()))
        path = tmp / f'e{ep}.npz'; store.write(path, dict(game=f'g{ep}', owner=0, outcome=1., opponent='current'))
        e = load(path); e['adv'] = gae(e['reward'], e['value'], .9997, .95); e['ret'] = e['adv'] + e['value']; episodes.append(e)
    x, act, ex, valid, t_max = collate(episodes, device)
    with torch.no_grad():
        logp, entropy, value = model.evaluate(x, act, t_max)
    dl = (logp - ex['logp'])[valid].abs(); dv = (value - ex['value'])[valid].abs()
    report = dict(steps=int(valid.sum()), acted=int((act['gate'] == 1)[valid].sum()), logp_max_abs_diff=float(dl.max()),
                  logp_mean_abs_diff=float(dl.mean()), value_max_abs_diff=float(dv.max()), entropy_mean=float(entropy[valid].mean()))
    print(json.dumps(report, indent=2))
    assert report['logp_max_abs_diff'] < 5e-2 and report['value_max_abs_diff'] < 5e-2, 'act/evaluate disagree'


if __name__ == '__main__':
    main()
