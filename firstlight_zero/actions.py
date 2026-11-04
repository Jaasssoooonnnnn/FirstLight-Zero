"""Policy choice (gate, option, model cell) -> native ActionV1, earliest legal delay."""
from native_runner.contracts import ActionKind, ActionV1, TargetKind

from .spec import BASE_LATENCY_TICKS, DECISION_TICKS, DELAY_BINS_MS


def wait_action(owner):
    return ActionV1.wait(owner, DECISION_TICKS)


def native_action(owner, featurizer, native, option, delay_bin, model_cell, action_id):
    """`native` is the per-option context returned by Featurizer.__call__."""
    entry = native[option]
    execute = BASE_LATENCY_TICKS + DELAY_BINS_MS[delay_bin] // 50
    meta = {'policy_action_schema': 'firstlight-zero.v1', 'policy_delay_offset_bin': int(delay_bin),
            'policy_delay_offset_ms': int(DELAY_BINS_MS[delay_bin]), 'execute_offset_ticks': execute,
            'base_latency_ticks': BASE_LATENCY_TICKS}
    if entry['kind'] == 'ability':
        return ActionV1(owner=owner, kind=ActionKind.ACTIVATE_ABILITY, source_entity=entry['source_entity'],
                        ability_id=entry['ability_id'], execute_offset_ticks=execute,
                        next_decision_ticks=DECISION_TICKS, action_id=action_id, metadata=meta)
    subcell = entry['subcell']
    if subcell is not None:
        # The mask reports the anchor offset in the seat's own frame; owner 1 is rotated.
        subcell = (-float(subcell[0]), -float(subcell[1])) if featurizer.rotate else (float(subcell[0]), float(subcell[1]))
    return ActionV1(owner=owner, kind=ActionKind.PLAY_CARD, hand_slot=entry['hand_slot'], card_id=entry['card_id'],
                    target_kind=TargetKind.GRID, target_grid=featurizer.native_cell(model_cell),
                    subcell_offset=subcell,
                    execute_offset_ticks=execute, next_decision_ticks=DECISION_TICKS,
                    action_id=action_id, metadata=meta)
