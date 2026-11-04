"""ObservationV1 -> compact owner-perspective arrays for the X-Bow mirror.

Owner 0 already sits at the bottom of the native board. Owner 1's view is
rotated 180 degrees so both seats see themselves at the bottom; legal cells
and the chosen cell use the same rotation.
"""
import collections
import math

import numpy as np

from native_runner.training.v4.native_actions import ability_runtime_contract

from .spec import (ABILITY_OPTION, CARD_INDEX, CELLS, DELAY_BINS_MS, ENTITY_FEATURES, ENTITY_INDEX,
                   ENTITY_KINDS, ENTITY_OTHER, EVO_CARDS, FORCE_OPEN_TICK, HEIGHT, MAX_ENTITIES, NO_CARD,
                   OPP_QUEUE, BASE_OPTION_FEATURES, OPTIONS, SCALAR_FEATURES, UNITS_X, UNITS_Y, WIDTH)

_KIND = {kind: i for i, kind in enumerate(ENTITY_KINDS)}


class Featurizer:
    """One per (game, owner). Tracks the public opponent cycle and elixir."""

    def __init__(self, owner, initial_obs, deck_layout):
        self.owner = owner
        self.rotate = owner == 1
        self.deck_layout = tuple(deck_layout)
        me = _player(initial_obs, owner)
        self.opp_elixir = float(me.elixir_exact)        # both seats start equal
        self.last_tick = int(initial_obs.tick)
        self.event_tick = -1
        self.seen = set()
        self.opp_queue = []
        self.opp_evo_plays = [0, 0]
        self.my_last_play_tick = None
        self.my_first_play_tick = None
        self.my_plays = []   # (tick, card_id, (x, y) in this seat's frame) of executed plays
        self.acted = False   # set by the driver once this seat submits any ACT; forced opening fires at most once
        self.unknown_entities = 0
        self.unknown_ids = collections.Counter()

    # ------------------------------------------------------------------ helpers
    def _xy(self, position):
        x, y = float(position[0]), float(position[1])
        if self.rotate:
            x, y = UNITS_X - x, UNITS_Y - y
        return x / UNITS_X, y / UNITS_Y

    def native_cell(self, model_cell):
        x, y = model_cell % WIDTH, model_cell // WIDTH
        if self.rotate:
            x, y = WIDTH - 1 - x, HEIGHT - 1 - y
        return x, y

    def _track_events(self, obs, regen_per_tick):
        tick = int(obs.tick)
        self.opp_elixir = min(10.0, self.opp_elixir + regen_per_tick * (tick - self.last_tick))
        self.last_tick = tick
        newest = self.event_tick
        for event in obs.events:
            if event.event_type != 'action_executed':
                continue
            event_tick = int(event.tick)
            if event_tick < self.event_tick:
                continue
            key = (event_tick, event.owner, event.card_id, tuple(event.position or ()))
            if key in self.seen:
                continue
            self.seen.add(key)
            newest = max(newest, event_tick)
            if event.owner == self.owner:
                self.my_plays.append((event_tick, event.card_id, self._xy(event.position) if event.position else None))
                self.my_last_play_tick = event_tick
                if self.my_first_play_tick is None:
                    self.my_first_play_tick = event_tick
                continue
            cost = (event.data or {}).get('effective_cost')
            if isinstance(cost, (int, float)):
                self.opp_elixir = max(0.0, self.opp_elixir - float(cost))
            card = CARD_INDEX.get(event.card_id, NO_CARD)
            if card != NO_CARD:
                self.opp_queue.append(card)
                del self.opp_queue[:-OPP_QUEUE]
                if event.card_id in EVO_CARDS:
                    self.opp_evo_plays[EVO_CARDS.index(event.card_id)] += 1
        if newest > self.event_tick:
            self.event_tick = newest
            self.seen = {key for key in self.seen if key[0] >= newest}

    # ------------------------------------------------------------------ main
    def __call__(self, obs):
        """Returns (inputs, legality, context). Arrays are float32/int64/bool numpy."""
        me = _player(obs, self.owner)
        mask = obs.action_mask
        reasons = mask.reasons or {}
        regen = float(reasons.get('elixir_regen_per_tick') or 0.0)
        self._track_events(obs, regen)
        tick = int(obs.tick)
        window = [float(v) for v in (reasons.get('touch_window_elixir') or ())] or [float(me.elixir_exact)] * len(DELAY_BINS_MS)

        # ---- options: four native hand slots + ability -------------------
        # me.hand is compacted while a slot waits for its refill; the placement
        # masks are keyed by native slot, so they define slot -> card.
        hand = [None] * 4
        hand_idx = np.full(5, NO_CARD, dtype=np.int64)
        for slot in range(4):
            entry = mask.placement_masks.get(str(slot))
            if entry is not None:
                hand[slot] = int(entry['card_id'])
                hand_idx[slot] = CARD_INDEX.get(hand[slot], NO_CARD)
        hand_idx[4] = CARD_INDEX.get(me.next_card, NO_CARD)

        evo = {state.card_id: state for state in me.evolution_runtime_states}
        opt_mask = np.zeros(OPTIONS, dtype=bool)
        opt_bin = np.zeros(OPTIONS, dtype=np.int64)
        opt_feat = np.zeros((OPTIONS, BASE_OPTION_FEATURES), dtype=np.float32)
        place = np.zeros((4, HEIGHT, WIDTH), dtype=bool)
        native = [None] * OPTIONS
        for slot, card in enumerate(hand):
            entry = mask.placement_masks.get(str(slot))
            if entry is None:
                continue
            cost = float(entry.get('effective_cost'))
            ready = evo.get(card)
            opt_feat[slot] = (0.0, cost / 10.0, 1.0 if ready is not None and ready.ready else 0.0, 0.0)
            if not mask.hand_slots[slot]:
                continue
            delay = _earliest_bin(window, cost)
            if delay is None:
                continue
            grid = np.asarray(entry['row_major'], dtype=bool)
            if self.rotate:
                grid = grid[::-1, ::-1]
            if not grid.any():
                continue
            place[slot] = grid
            opt_mask[slot] = True
            opt_bin[slot] = delay
            opt_feat[slot, 0] = 1.0
            opt_feat[slot, 3] = delay / 4.0
            native[slot] = dict(kind='card', hand_slot=slot, card_id=int(card),
                                subcell=entry.get('model_subcell_offset'), cost=cost)
        hand_idx_ability = NO_CARD
        for source in mask.ability_sources:
            contract = ability_runtime_contract(obs, source_entity=int(source), deck=self.deck_layout)
            delay = _earliest_bin(window, float(contract.cost))
            hand_idx_ability = CARD_INDEX.get(contract.source_card_id, NO_CARD)
            opt_feat[ABILITY_OPTION, 1] = float(contract.cost) / 10.0
            if delay is None:
                continue
            opt_mask[ABILITY_OPTION] = True
            opt_bin[ABILITY_OPTION] = delay
            opt_feat[ABILITY_OPTION, 0] = 1.0
            opt_feat[ABILITY_OPTION, 3] = delay / 4.0
            native[ABILITY_OPTION] = dict(kind='ability', source_entity=int(source),
                                          ability_id=contract.ability_id, cost=float(contract.cost))
            break
        option_card = np.concatenate([hand_idx[:4], [hand_idx_ability]])

        # ---- scalars -------------------------------------------------------
        opp = _player(obs, 1 - self.owner)
        towers = {(t.owner, _tower_side(t, self.rotate)): t for t in obs.towers}
        scal = np.zeros(SCALAR_FEATURES, dtype=np.float32)
        scal[0] = tick / 6000.0
        scal[1] = float(obs.time.remaining_ms or 0) / 180000.0
        scal[2] = float(obs.time.elixir_multiplier or 1.0) / 3.0
        scal[3] = 0.0 if obs.phase == 'normal' else 1.0
        scal[4] = float(me.elixir_exact) / 10.0
        scal[5] = self.opp_elixir / 10.0
        scal[6] = me.crowns / 3.0
        scal[7] = opp.crowns / 3.0
        for i, key in enumerate(((self.owner, 'left'), (self.owner, 'right'), (self.owner, 'king'),
                                 (1 - self.owner, 'left'), (1 - self.owner, 'right'), (1 - self.owner, 'king'))):
            tower = towers.get(key)
            if tower is not None and tower.hitpoints:
                scal[8 + i] = float(tower.hitpoints) / float(tower.max_hitpoints or 1.0)
        king_me, king_opp = towers.get((self.owner, 'king')), towers.get((1 - self.owner, 'king'))
        scal[14] = 1.0 if king_me is not None and king_me.active else 0.0
        scal[15] = 1.0 if king_opp is not None and king_opp.active else 0.0
        for i, card in enumerate(EVO_CARDS):
            state = evo.get(card)
            if state is not None:
                scal[16 + 2 * i] = float(state.cycle_remaining or 0) / 2.0
                scal[17 + 2 * i] = 1.0 if state.ready else 0.0
            scal[20 + i] = (self.opp_evo_plays[i] % 3) / 2.0
        scal[22] = 1.0 if mask.ability_sources else 0.0
        scal[23] = min(20.0, (tick - self.my_last_play_tick) / 20.0) / 20.0 if self.my_last_play_tick is not None else 1.0
        scal[24] = 1.0 if self.my_first_play_tick is not None else 0.0
        scal[25] = float(window[0]) / 10.0

        opp_queue = np.full(OPP_QUEUE, NO_CARD, dtype=np.int64)
        opp_queue[OPP_QUEUE - len(self.opp_queue):] = self.opp_queue

        # ---- entities ------------------------------------------------------
        rows = []
        for entity in obs.entities:
            kind = _KIND.get(entity.entity_kind, len(ENTITY_KINDS))
            type_id = ENTITY_INDEX.get(entity.card_id, ENTITY_OTHER)
            if type_id == ENTITY_OTHER:
                self.unknown_entities += 1
                self.unknown_ids[(entity.card_id, entity.entity_kind)] += 1
            x, y = self._xy(entity.position)
            vx, vy = (entity.velocity or (0.0, 0.0))[:2]
            if self.rotate:
                vx, vy = -vx, -vy
            hp, max_hp = entity.hitpoints, entity.max_hitpoints
            hp_frac = float(hp) / float(max_hp) if hp is not None and max_hp else 1.0
            feat = [1.0 if entity.owner == self.owner else 0.0, x, y, hp_frac,
                    math.log1p(float(max_hp or 0.0)) / 8.0,
                    float(entity.shield or 0.0) / max(1.0, float(max_hp or 1.0)),
                    min(1.0, float(entity.age_ms or 0) / 10000.0),
                    float(vx) / 100.0, float(vy) / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            feat[9 + kind] = 1.0
            rows.append((type_id, feat))
            if len(rows) == MAX_ENTITIES:
                break
        ent_type = np.zeros(MAX_ENTITIES, dtype=np.int64)
        ent_feat = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
        for i, (type_id, feat) in enumerate(rows):
            ent_type[i] = type_id
            ent_feat[i] = feat

        force = tick >= FORCE_OPEN_TICK and not self.acted and self.my_first_play_tick is None and bool(opt_mask.any())
        inputs = dict(scal=scal, hand=hand_idx, option_card=option_card, opt_feat=opt_feat, opp_queue=opp_queue,
                      ent_type=ent_type, ent_feat=ent_feat, ent_n=np.int64(len(rows)))
        legality = dict(opt_mask=opt_mask, place=place, opt_bin=opt_bin, force=force)
        return inputs, legality, native


def _player(obs, owner):
    return next(p for p in obs.players if p.owner == owner)


def _tower_side(tower, rotate):
    kind = str(tower.tower_kind)
    if kind == 'king':
        return 'king'
    left = float(tower.position[0]) < UNITS_X / 2
    return 'left' if left != rotate else 'right'


def _earliest_bin(window, cost):
    for i, value in enumerate(window):
        if value + 1e-6 >= cost:
            return i
    return None
