"""Stack per-seat numpy inputs/legality into model tensors (shared by the inference server and the evaluator)."""
import numpy as np
import torch

from .spec import HEIGHT, WIDTH

LONG_KEYS = ('unit_type', 'option_card')
FLOAT_KEYS = ('unit_feat', 'scal', 'cards', 'opp_cards', 'opt_feat')
WIRE_DTYPES = dict(unit_type=np.int16, option_card=np.int8, unit_feat=np.float16, scal=np.float16, cards=np.float16, opp_cards=np.float16, opt_feat=np.float16)


def stack_inputs(rows, device):
    x = {}
    for k in rows[0]:
        if k == 'unit_n':
            continue
        arr = torch.from_numpy(np.stack([r[k] for r in rows]))
        x[k] = (arr.long() if k in LONG_KEYS else arr.float()).to(device, non_blocking=True)
    return x


def pack_place(place):
    return np.packbits(place.reshape(4, -1), axis=-1)          # [4, 72] uint8


def unpack_place_bits(bits):
    return np.unpackbits(bits, axis=-1)[..., :HEIGHT * WIDTH].reshape(bits.shape[0], 4, HEIGHT, WIDTH).astype(bool)


def stack_legality(rows, device):
    return dict(opt_mask=torch.from_numpy(np.stack([r['opt_mask'] for r in rows])).to(device),
                place=torch.from_numpy(np.stack([r['place'] for r in rows])).to(device),
                force=torch.tensor([bool(r['force']) for r in rows], device=device))


def pack_request(client, seats, greedy=False, temps=None):
    """seats: list of (inputs, legality, model_key, state_key, start). Arrays are stacked per field and sent compactly
    (float16 / small ints), so the server unpickles a handful of arrays per request instead of one object per seat.
    `temps` (eval only): per-seat sampling temperature, 0 = argmax."""
    inputs = {k: np.stack([s[0][k] for s in seats]).astype(WIRE_DTYPES[k]) for k in WIRE_DTYPES}
    extra = {} if temps is None else dict(temp=np.asarray(temps, np.float32))
    return dict(client=client, n=len(seats), model=[s[2] for s in seats], greedy=greedy, **extra,
                key=np.array([s[3] for s in seats], np.int32), start=np.array([s[4] for s in seats], bool), inputs=inputs,
                opt_mask=np.stack([s[1]['opt_mask'] for s in seats]), place_bits=np.stack([pack_place(s[1]['place']) for s in seats]),
                force=np.array([bool(s[1]['force']) for s in seats]))
