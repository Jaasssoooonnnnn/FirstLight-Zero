"""Frozen V4 opponents (imitation-learning checkpoints and their PPO post-trainings): argmax, earliest delay, 200 ms ceiling."""
from contextvars import ContextVar
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

from native_runner.training.v4 import model as _v4_model   # noqa: E402
from native_runner.training.v4.policy_loader import load_policy_v4   # noqa: E402

DELAY_CEILING_MS = 200

# The decoder ignores its runtime delay cap during teacher forcing; the forced-opening path must score the same capped
# distribution that argmax used, so the cap is applied through this subclass.
_runtime_cap = ContextVar('zero_v4_runtime_cap', default=None)
_NativeShadow = _v4_model.ShadowCandidateLegality


class _CappedShadow(_NativeShadow):
    def __init__(self, *args, allowed_delay_bins=None, **kwargs):
        if allowed_delay_bins is None:
            allowed_delay_bins = _runtime_cap.get()
        super().__init__(*args, allowed_delay_bins=allowed_delay_bins, **kwargs)


_v4_model.ShadowCandidateLegality = _CappedShadow


class ConsistentPolicy(_v4_model.UniversalCardPolicyV4):
    def _decode(self, *args, **kwargs):
        token = _runtime_cap.set(self._inference_delay_offset_mask)
        try:
            return super()._decode(*args, **kwargs)
        finally:
            _runtime_cap.reset(token)

    def _evaluate_sparse_forced_actions(self, *args, **kwargs):
        token = _runtime_cap.set(self._inference_delay_offset_mask)
        try:
            return super()._evaluate_sparse_forced_actions(*args, **kwargs)
        finally:
            _runtime_cap.reset(token)


def load_opponent(path, device, learner=False):
    assert not learner, 'opponents are frozen'
    model = load_policy_v4(path, device=device, delay_policy='earliest').model
    model.set_inference_delay_offset_max_ms(DELAY_CEILING_MS)
    model.__class__ = ConsistentPolicy
    model.set_ppo_location_head_fp32(True)
    # Argmax scores its chosen heads at unit temperature; PPO checkpoints retain their training temperatures.
    model.set_ppo_gate_temperature(1.)
    model.set_ppo_action_temperature(1.)
    model.set_ppo_continue_temperature(1.)
    model.set_ppo_rollout_sampling(False)
    return model
