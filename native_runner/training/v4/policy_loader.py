"""Loads frozen V4 checkpoints used as evaluation opponents."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
from pathlib import Path

import torch

from .checkpoint import model_config_from_contract, read_actor_critic_checkpoint, restore_actor_critic_checkpoint
from .decoding import DELAY_POLICY_SAMPLE
from .factory import build_production_model_v4
from .model import UniversalCardPolicyV4
from .tensors import (
    ActiveEffectSetV4,
    PolicyOutputV4,
    RelationEdgesV4,
    RecurrentPolicyStateV4,
    UniversalSemanticBatchV4,
)


_CUDA_GRAPH_EFFECT_CAPACITY = 512
_CUDA_GRAPH_RELATION_CAPACITY = 2048
_DELAY_OFFSET_MAX_MS = 200


@dataclass(frozen=True, slots=True)
class LoadedPolicyV4:
    model: UniversalCardPolicyV4
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_id: str
    update_step: int
    training_stage: str


def _copy_tensor_record_(target: object, source: object) -> None:
    """Copy matching nested tensor records without allocating new CUDA inputs."""

    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError("V4 tensor record source is not a tensor")
        if target.shape != source.shape:
            raise ValueError("V4 CUDA Graph input shape changed")
        target.copy_(source)
        return
    if is_dataclass(target):
        if type(source) is not type(target):
            raise TypeError("V4 CUDA Graph tensor record type changed")
        for field in fields(target):
            _copy_tensor_record_(getattr(target, field.name), getattr(source, field.name))
        return
    if target != source:
        raise ValueError("V4 CUDA Graph non-tensor input changed")


def _pad_axis_one(value: torch.Tensor, capacity: int, pad_value: int | float | bool) -> torch.Tensor:
    count = int(value.shape[1])
    if count > capacity:
        raise ValueError(f"V4 CUDA Graph capacity exceeded: {count} > {capacity}")
    if count == capacity:
        return value
    result = torch.full((value.shape[0], capacity, *value.shape[2:]), pad_value, dtype=value.dtype, device=value.device)
    if count:
        result[:, :count].copy_(value)
    return result


def _fixed_cuda_graph_batch_v4(
    batch: UniversalSemanticBatchV4,
    *,
    effect_capacity: int = _CUDA_GRAPH_EFFECT_CAPACITY,
    relation_capacity: int = _CUDA_GRAPH_RELATION_CAPACITY,
) -> UniversalSemanticBatchV4:
    """Pad the two intentionally variable V4 sets for one reusable graph."""

    if effect_capacity <= 0 or relation_capacity <= 0:
        raise ValueError("V4 CUDA Graph capacities must be positive")

    effects = batch.active_effects
    fixed_effects = ActiveEffectSetV4(
        effect_vocab_id=_pad_axis_one(effects.effect_vocab_id, effect_capacity, 0),
        parent_type=_pad_axis_one(effects.parent_type, effect_capacity, 0),
        parent_index=_pad_axis_one(effects.parent_index, effect_capacity, -1),
        source_owner_type=_pad_axis_one(effects.source_owner_type, effect_capacity, 0),
        runtime_features=_pad_axis_one(effects.runtime_features, effect_capacity, 0.0),
        mask=_pad_axis_one(effects.mask, effect_capacity, False),
    )
    relations = batch.relation_edges
    fixed_relations = RelationEdgesV4(
        source=_pad_axis_one(relations.source, relation_capacity, 0),
        target=_pad_axis_one(relations.target, relation_capacity, 0),
        relation_type=_pad_axis_one(relations.relation_type, relation_capacity, 0),
        mask=_pad_axis_one(relations.mask, relation_capacity, False),
    )
    return replace(batch, active_effects=fixed_effects, relation_edges=fixed_relations)


class _DeterministicCudaGraphV4:
    """One reusable batch-1 deterministic graph owned by a loaded model."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.batch: UniversalSemanticBatchV4 | None = None
        self.state: RecurrentPolicyStateV4 | None = None
        self.episode_start: torch.Tensor | None = None
        self.output: PolicyOutputV4 | None = None

    def _capture(
        self,
        model: UniversalCardPolicyV4,
        host_batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: torch.Tensor,
    ) -> PolicyOutputV4:
        fixed_batch = _fixed_cuda_graph_batch_v4(host_batch)
        self.batch = fixed_batch.to_model_input(self.device)
        self.state = RecurrentPolicyStateV4(hidden=state.hidden.detach().clone(), cell=state.cell.detach().clone())
        self.episode_start = episode_start.detach().clone()

        # CUDA Graph capture requires allocations/kernels to be warmed on a
        # side stream. The captured graph then owns stable input/output storage
        # and can be reused across episodes by copying new values in-place.
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                model.act(self.batch, self.state, episode_start=self.episode_start, validate=False)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model.act(self.batch, self.state, episode_start=self.episode_start, validate=False)
        if self.output is None:
            raise RuntimeError("V4 CUDA Graph capture produced no output")
        # Replay once before exposing the static output tensors. Capture builds
        # the executable graph and reserves its pool; only replay is treated as
        # the authoritative inference execution across CUDA/PyTorch versions.
        self.graph.replay()
        return self.output

    def run(
        self,
        model: UniversalCardPolicyV4,
        host_batch: UniversalSemanticBatchV4,
        state: RecurrentPolicyStateV4,
        episode_start: torch.Tensor,
    ) -> PolicyOutputV4:
        if self.graph is None:
            return self._capture(model, host_batch, state, episode_start)
        if self.batch is None or self.state is None or self.episode_start is None or self.output is None:
            raise RuntimeError("V4 CUDA Graph storage is incomplete")
        _copy_tensor_record_(self.batch, _fixed_cuda_graph_batch_v4(host_batch))
        _copy_tensor_record_(self.state, state)
        self.episode_start.copy_(episode_start)
        self.graph.replay()
        return self.output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_v4_checkpoint(path: str | Path) -> Path:
    """Resolve one exact V4 checkpoint, accepting a training directory."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"V4 checkpoint does not exist: {candidate}")
    matches = sorted(candidate.glob("checkpoint-step-*.pt"))
    if not matches:
        matches = sorted(candidate.glob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"checkpoint directory contains no V4 .pt files: {candidate}")
    return matches[-1].resolve()


def load_policy_v4(
    checkpoint: str | Path, *, device: torch.device | str, delay_policy: str = DELAY_POLICY_SAMPLE
) -> LoadedPolicyV4:
    """Construct the V4 model from the checkpoint's contract and load it strictly."""

    checkpoint_path = resolve_v4_checkpoint(checkpoint)
    actual_device = torch.device(device)
    payload = read_actor_critic_checkpoint(checkpoint_path, map_location=actual_device)
    model = build_production_model_v4(device=actual_device, config=model_config_from_contract(payload["contract"]))
    payload = restore_actor_critic_checkpoint(payload, model, restore_rng=False)
    if model.config.delay_aware_legality:
        model.set_inference_delay_offset_max_ms(_DELAY_OFFSET_MAX_MS)
    model.set_delay_policy(delay_policy)
    model.eval()
    return LoadedPolicyV4(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_id=str(payload["checkpoint_id"]),
        update_step=int(payload["update_step"]),
        training_stage=str(payload["training_stage"]),
    )
