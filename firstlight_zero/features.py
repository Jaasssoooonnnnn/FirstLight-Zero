"""ObservationV1 -> the pruned-IL input schema (firstlight_zero.spec), owner perspective (owner 1 rotated 180 degrees).

Legality, native option context and event bookkeeping come from firstlight_zero.legality.Featurizer (validated against the
original decoder). The public opponent state (cycle, elixir interval, evolution, hero ability) comes from the
original V4 tensorizer's DeterministicPublicTracker, which this class drives directly.
"""
import numpy as np

from native_runner.contracts import ActionKind, AttackPhase, ProjectilePhase
from native_runner.training.tracking import AvailabilityIndex
from native_runner.training.v4.factory import build_episode_tensorizer_v4

from firstlight_zero.legality import Featurizer as BaseFeaturizer, _player, _tower_side
from .spec import (ABILITY_OPTION, ARCHERS, CARDS, CARD_FEATURES, KNIGHT, MAX_UNITS, OPP_CARD_FEATURES, OPTIONS,
                   OPTION_FEATURES, SCALARS, TESLA, UNIT_FEATURES, UNIT_INDEX, UNIT_KING, UNIT_OTHER, UNIT_PRINCESS,
                   UNITS_X, UNITS_Y, WIDTH, HEIGHT)

UNIT_HERO = UNIT_INDEX['form:KnightHero']


class Featurizer2:
    def __init__(self, owner, episode, initial_obs, deck_layout):
        self.owner = owner; self.enemy = 1 - owner
        self.base = BaseFeaturizer(owner, initial_obs, deck_layout)
        self.rotate = self.base.rotate
        self.tz = build_episode_tensorizer_v4(episode, actor_owner=owner)
        mine = float(_player(initial_obs, owner).elixir_exact)   # the opponent's exact elixir is private; both seats start equal
        elixir = {0: mine, 1: mine}
        self.tz.start_episode(initial_obs, initial_elixir=elixir if self.tz.tracker is not None else None)
        catalog = self.tz.entity_archetype_catalog
        self.unit_index = {}
        for vid in range(catalog.vocab_size):
            key = catalog.metadata_for_vocab_id(vid).archetype_key
            if key in UNIT_INDEX:
                self.unit_index[vid] = UNIT_INDEX[key]
        self.other_units = 0

    # convenience passthroughs used by the drivers
    @property
    def acted(self): return self.base.acted

    @acted.setter
    def acted(self, value): self.base.acted = value

    @property
    def my_plays(self): return self.base.my_plays

    def native_cell(self, cell): return self.base.native_cell(cell)

    def _xy(self, position):
        return self.base._xy(position)

    def __call__(self, obs):
        inputs0, legality, native = self.base(obs)
        tracker = self.tz.tracker
        if tracker is not None:
            tracker.update(obs); tracker.reconcile_owner_private_state(obs)
        me, opp = _player(obs, self.owner), _player(obs, self.enemy)
        tick = int(obs.tick); reasons = obs.action_mask.reasons or {}

        # ---- units (towers included) ----------------------------------------
        self.tz._frame_archetype_id_cache.clear(); self.tz._frame_child_kind_cache.clear()
        unit_type = np.zeros(MAX_UNITS, np.int64); unit_feat = np.zeros((MAX_UNITS, UNIT_FEATURES), np.float32); n = 0
        opp_hero_on_board = False
        for tower in obs.towers:
            x, y = self._xy(tower.position)
            unit_type[n] = UNIT_KING if str(tower.tower_kind) == 'king' else UNIT_PRINCESS
            f = unit_feat[n]; f[0] = tower.owner == self.owner; f[1] = x; f[2] = y
            f[3] = float(tower.hitpoints or 0) / float(tower.max_hitpoints or 1); f[4] = float(tower.hitpoints or 0) / 5000.
            f[6] = tower.visible_target is not None
            n += 1
        for e in obs.entities:
            if e.owner is None or n >= MAX_UNITS:
                continue
            if self.tz._entity_child_kind(e) == 'area':
                continue
            t = self.unit_index.get(self.tz._entity_archetype_id(e), UNIT_OTHER)
            if t == UNIT_OTHER: self.other_units += 1
            if t == UNIT_HERO and e.owner != self.owner: opp_hero_on_board = True
            unit_type[n] = t; f = unit_feat[n]
            x, y = self._xy(e.position); f[0] = e.owner == self.owner; f[1] = x; f[2] = y
            hp, mhp = e.hitpoints, e.max_hitpoints
            f[3] = float(hp) / float(mhp) if hp is not None and mhp else 0.; f[4] = float(hp or 0) / 5000.
            f[5] = min(1., float(e.age_ms or 0) / 60000.); f[6] = e.visible_target is not None
            if e.velocity is not None:
                vx, vy = float(e.velocity[0]), float(e.velocity[1])
                if self.rotate: vx, vy = -vx, -vy
                f[7] = vx / 1000.; f[8] = vy / 1000.     # world units per tick -> tiles per tick is /1000
            a = e.attack_state
            if a is not None:
                f[9] = min(1., float(a.cooldown_remaining_ms or 0) / 5000.)
                f[10] = a.phase == AttackPhase.WINDUP; f[11] = a.phase == AttackPhase.COOLDOWN
            d = e.deployment_runtime
            if d is not None:
                f[12] = d.get('phase') == 'deploying'; f[13] = min(1., float(d.get('remaining_native_ms') or 0) / 5000.)
            p = e.projectile_state
            if p is not None:
                f[14] = p.phase == ProjectilePhase.IN_FLIGHT
                if p.target_position is not None:
                    f[15], f[16] = self._xy(p.target_position)
            n += 1

        # ---- my cards / opponent cards (CARDS order) -------------------------
        cards = np.zeros((8, CARD_FEATURES), np.float32); opp_cards = np.zeros((8, OPP_CARD_FEATURES), np.float32)
        pending = {}
        for item in obs.pending_actions:
            act = item.action
            if act.kind is ActionKind.PLAY_CARD and act.target_grid is not None:
                gx, gy = act.target_grid   # native grid; rotate for owner 1
                if self.rotate: gx, gy = WIDTH - 1 - gx, HEIGHT - 1 - gy
                pending[int(act.card_id)] = ((gx + .5) / WIDTH, (gy + .5) / HEIGHT)
        for i, card in enumerate(CARDS):
            cards[i, 0] = card in me.hand; cards[i, 1] = me.next_card == card
            if card in pending:
                cards[i, 2] = 1.; cards[i, 3], cards[i, 4] = pending[card]
            if tracker is not None:
                try:
                    st = tracker.card_state(self.enemy, card)
                except (KeyError, ValueError):
                    st = None
                if st is not None:
                    opp_cards[i, 0] = st.availability == AvailabilityIndex.AVAILABLE
                    opp_cards[i, 1] = st.availability == AvailabilityIndex.CYCLING
                    opp_cards[i, 2] = min(1., float(st.cycle_distance) / 4.)

        # ---- options --------------------------------------------------------
        opt_feat = np.zeros((OPTIONS, OPTION_FEATURES), np.float32)
        opt_feat[:, 0] = inputs0['opt_feat'][:, 1]                      # cost / 10
        for o in range(4):
            if legality['opt_mask'][o]: opt_feat[o, 1] = legality['place'][o].mean()
        opt_feat[ABILITY_OPTION, 1] = float(legality['opt_mask'][ABILITY_OPTION])
        opt_feat[:, 2] = legality['opt_bin'] / 4.

        # ---- scalars --------------------------------------------------------
        s = np.zeros(SCALARS, np.float32)
        s[0] = float(obs.time.remaining_ms or 0) / 300000.; s[1] = float(obs.time.elixir_multiplier or 1.) / 3.
        s[2] = obs.phase in ('overtime', 'sudden_death'); s[3] = obs.phase == 'tiebreak'
        s[4] = float(me.elixir_exact) / 10.; s[5] = float(reasons.get('reserved_elixir') or 0.) / 10.
        # deterministic estimate from executed plays (legality.py), plus the public tracker's upper bound
        s[6] = self.base.opp_elixir / 10.
        s[7] = tracker.elixir_interval(self.enemy)[1] / 10. if tracker is not None else s[6]
        s[8] = me.crowns / 3.; s[9] = opp.crowns / 3.
        towers = {(t.owner, _tower_side(t, self.rotate)): t for t in obs.towers}
        for i, key in enumerate(((self.owner, 'left'), (self.owner, 'right'), (self.owner, 'king'),
                                 (self.enemy, 'left'), (self.enemy, 'right'), (self.enemy, 'king'))):
            t = towers.get(key)
            if t is not None and t.hitpoints: s[10 + i] = float(t.hitpoints) / float(t.max_hitpoints or 1.)
        for i, o in enumerate((self.owner, self.enemy)):
            t = towers.get((o, 'king')); s[16 + i] = t is not None and 'activated' in t.status
        evo = {st.card_id: st for st in me.evolution_runtime_states}
        for j, card in enumerate((ARCHERS, TESLA)):
            st = evo.get(card)
            if st is not None:
                s[18 + 2 * j] = st.ready is True; s[19 + 2 * j] = min(1., float(st.cycle_remaining or 0) / 4.)
            if tracker is not None:
                try:
                    pub = tracker.card_state(self.enemy, card)
                    s[23 + 2 * j] = pub.evolution_ready; s[24 + 2 * j] = min(1., float(pub.evolution_cycle_remaining) / 4.)
                except (KeyError, ValueError):
                    pass
        s[22] = bool(obs.action_mask.ability_sources) or any(st.available is True for st in me.ability_runtime_states)
        if tracker is not None and self.tz.ability_ids_by_card.get(KNIGHT):
            try:
                s[27] = opp_hero_on_board and tracker.ability_state(self.enemy, tick=tick, card_id=KNIGHT).ready
            except (KeyError, ValueError, AttributeError):
                pass
        s[28] = self.base.my_first_play_tick is not None
        if tracker is not None and self.tz.ability_ids_by_card.get(KNIGHT):
            for i, o in enumerate((self.owner, self.enemy)):
                try:
                    activated = tracker.ability_state(o, tick=tick, card_id=KNIGHT).activated_tick
                except (KeyError, ValueError, AttributeError):
                    activated = None
                if activated is not None:
                    s[29 + i] = max(0., 1. - (tick - int(activated)) / 100.)      # 5 s = 100 ticks

        inputs = dict(scal=s, cards=cards, opp_cards=opp_cards, option_card=inputs0['option_card'], opt_feat=opt_feat,
                      unit_type=unit_type, unit_feat=unit_feat, unit_n=np.int64(n))
        return inputs, legality, native
