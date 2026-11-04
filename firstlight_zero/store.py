"""Per-seat trajectory buffer / .npz format for the v2 schema (units ragged, legal cells bit-packed for card plays)."""
import json
import os

import numpy as np

from .spec import ABILITY_OPTION, HEIGHT, MAX_UNITS, UNIT_FEATURES, WIDTH

FIXED = ('scal', 'cards', 'opp_cards', 'option_card', 'opt_feat')
DTYPES = dict(scal=np.float16, cards=np.float16, opp_cards=np.float16, option_card=np.int8, opt_feat=np.float16)
FLOAT_KEYS = ('scal', 'cards', 'opp_cards', 'opt_feat')


class SeatStore:
    def __init__(self):
        self.rows = {k: [] for k in FIXED}
        self.unit_n = []; self.unit_type = []; self.unit_feat = []
        self.opt_mask = []; self.gate = []; self.option = []; self.cell = []; self.gate_free = []
        self.logp = []; self.value = []; self.version = []; self.rewards = []
        self.place_index = []; self.place_bits = []

    def append(self, inputs, legality, decision, version):
        for k in FIXED:
            self.rows[k].append(inputs[k])
        n = int(inputs['unit_n'])
        self.unit_n.append(n); self.unit_type.append(inputs['unit_type'][:n]); self.unit_feat.append(inputs['unit_feat'][:n])
        self.opt_mask.append(legality['opt_mask'])
        gate, option = int(decision['gate']), int(decision['option'])
        self.gate.append(gate); self.option.append(option); self.cell.append(int(decision['cell']))
        self.gate_free.append(bool(decision['gate_free'])); self.logp.append(float(decision['logp']))
        self.value.append(float(decision['value'])); self.version.append(int(version))
        if gate == 1 and option != ABILITY_OPTION:
            self.place_index.append(len(self.gate) - 1)
            self.place_bits.append(np.packbits(legality['place'][option].reshape(-1)))

    def reward(self, value, replace=False):
        if replace:
            self.rewards[-1] = float(value)
        else:
            self.rewards.append(float(value))

    def write(self, path, meta):
        if len(self.rewards) != len(self.gate):
            raise RuntimeError(f'reward/decision mismatch {len(self.rewards)} != {len(self.gate)}')
        arrays = {k: np.asarray(v, dtype=DTYPES[k]) for k, v in self.rows.items()}
        arrays.update(
            unit_n=np.asarray(self.unit_n, np.int8),
            unit_type=np.concatenate(self.unit_type).astype(np.int8) if self.unit_type else np.zeros(0, np.int8),
            unit_feat=np.concatenate(self.unit_feat).astype(np.float16) if self.unit_feat else np.zeros((0, UNIT_FEATURES), np.float16),
            opt_mask=np.asarray(self.opt_mask, bool), gate=np.asarray(self.gate, np.int8),
            option=np.asarray(self.option, np.int8), cell=np.asarray(self.cell, np.int16),
            gate_free=np.asarray(self.gate_free, bool), logp=np.asarray(self.logp, np.float32),
            value=np.asarray(self.value, np.float32), version=np.asarray(self.version, np.int32),
            reward=np.asarray(self.rewards, np.float32),
            place_index=np.asarray(self.place_index, np.int32),
            place_bits=np.asarray(self.place_bits, np.uint8).reshape(-1, (HEIGHT * WIDTH + 7) // 8),
            meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'wb') as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)


def load(path):
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data['meta'] = json.loads(bytes(data['meta']).decode())
    return data


def unpack_units(data, width=MAX_UNITS):
    """Ragged units -> dense [T, width] arrays."""
    t = len(data['unit_n'])
    unit_type = np.zeros((t, width), np.int64); unit_feat = np.zeros((t, width, UNIT_FEATURES), np.float32)
    counts = data['unit_n'].astype(np.int64)
    rows = np.repeat(np.arange(t), counts)
    cols = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    unit_type[rows, cols] = data['unit_type']; unit_feat[rows, cols] = data['unit_feat']
    return unit_type, unit_feat


def unpack_place(data):
    t = len(data['gate'])
    place = np.zeros((t, HEIGHT, WIDTH), bool)
    if len(data['place_index']):
        bits = np.unpackbits(data['place_bits'], axis=1)[:, :HEIGHT * WIDTH].astype(bool)
        place[data['place_index']] = bits.reshape(-1, HEIGHT, WIDTH)
    return place
