"""Immutable, content-addressed ruleset manifests.

A trajectory is only comparable with another trajectory when engine code,
public card definitions, observation/action contracts, timing/RNG semantics,
mode data, and latency rules agree.  RulesetManifestV1 makes that boundary
explicit and verifies every local file that contributed to the identity.
"""

from __future__ import annotations

from .paths import PACKAGE_ROOT, WORKSPACE_ROOT

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import struct
from typing import Any

from .card_logic import StaticCardLogicCatalogV1, build_static_card_logic_catalog
from .card_specs import CardSpecCatalog, build_card_catalog, sha256_file
from .effect_catalog import build_effect_catalog
from .projectile_catalog import build_projectile_catalog
from .semantic_subset import build_semantic_supported_subset, load_normal_mode_policy_readiness
from .runtime_scope import NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION, build_normal_mode_runtime_policy_scope
from .timeline import load_game_mode_timeline
from .contracts import (
    ACTION_VERSION,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_TICK_MS,
    LATENCY_CONFIG_VERSION,
    OBSERVATION_VERSION,
    ContractError,
    ContractMixin,
    EnvironmentConfigV1,
    FrozenMapping,
    content_hash,
    contract_schema_descriptor,
    frozen_mapping,
)


RULESET_MANIFEST_VERSION = "ruleset-manifest.v1"
RUNTIME_MATCH_LEVEL_POLICY_VERSION = "normalized-level-11.v1"
RUNTIME_MATCH_LEVEL_CAP = 11
RUNTIME_MATCH_MINIMUM_CARD_LEVEL = 11
RUNTIME_MATCH_KING_TOWER_LEVEL = 11
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _natural_version(path: Path) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", path.name))


def _relative(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


def _workspace_relative_source(value: str | Path, workspace: Path) -> str:
    """Normalize identity-bearing source paths without binding a checkout root."""

    path = Path(value)
    if not path.is_absolute():
        path = workspace / path
    return _relative(path, workspace)


def _hash_paths(paths: Iterable[Path], workspace: Path) -> FrozenMapping:
    result: dict[str, str] = {}
    for path in sorted({item.resolve() for item in paths if item.is_file()}, key=lambda item: item.as_posix()):
        result[_relative(path, workspace)] = sha256_file(path)
    return frozen_mapping(result)


def _component_hash(files: Mapping[str, str], descriptor: Mapping[str, Any] | None = None) -> str:
    return content_hash({"files": files, "descriptor": descriptor or {}})


def _read_patch_version(workspace: Path) -> str:
    fingerprint = workspace / "runtime-update" / "fingerprint.json"
    if fingerprint.is_file():
        try:
            data = json.loads(fingerprint.read_text(encoding="utf-8"))
            if data.get("version"):
                return str(data["version"])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    releases = sorted(workspace.glob("nr_*_release_*"), key=_natural_version, reverse=True)
    return releases[0].name if releases else "unknown"


def _read_runtime_fingerprint(workspace: Path) -> tuple[str, str]:
    fingerprint = workspace / "runtime-update" / "fingerprint.json"
    try:
        data = json.loads(fingerprint.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "", ""
    version = str(data.get("version", ""))
    manifest_sha1 = str(data.get("sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", manifest_sha1):
        manifest_sha1 = ""
    return version, manifest_sha1


def _elf_gnu_build_id(path: Path) -> str:
    """Read one ELF64 little-endian GNU build ID without external tooling."""

    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
        return ""
    try:
        program_offset = struct.unpack_from("<Q", data, 32)[0]
        entry_size = struct.unpack_from("<H", data, 54)[0]
        entry_count = struct.unpack_from("<H", data, 56)[0]
        for index in range(entry_count):
            offset = program_offset + index * entry_size
            if offset + 56 > len(data):
                return ""
            header = struct.unpack_from("<IIQQQQQQ", data, offset)
            if header[0] != 4:  # PT_NOTE
                continue
            note_offset, note_size = int(header[2]), int(header[5])
            cursor = note_offset
            end = min(len(data), note_offset + note_size)
            while cursor + 12 <= end:
                name_size, description_size, note_type = struct.unpack_from("<III", data, cursor)
                cursor += 12
                name = data[cursor : cursor + name_size]
                cursor += (name_size + 3) & ~3
                description = data[cursor : cursor + description_size]
                cursor += (description_size + 3) & ~3
                if note_type == 3 and name.rstrip(b"\0") == b"GNU":
                    return description.hex()
    except (OverflowError, struct.error):
        return ""
    return ""


def _discover_engine_files(workspace: Path) -> list[Path]:
    candidates = [
        item / "lib" / "arm64-v8a" / "libg.so"
        for item in sorted(workspace.glob("nr_*_release_*"), key=_natural_version, reverse=True)
    ]
    return (
        [next((item for item in candidates if item.is_file()), Path())]
        if any(item.is_file() for item in candidates)
        else []
    )


def _iter_files(root: Path, suffixes: set[str] | None = None) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return (
        path for path in root.rglob("*") if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes)
    )


@dataclass(frozen=True, slots=True)
class RulesetManifestV1(ContractMixin):
    ruleset_version: str
    engine_build_hash: str
    simulator_build_hash: str
    card_schema_hash: str
    observation_contract_hash: str
    action_contract_hash: str
    tick_rng_latency_rules_hash: str
    card_specs_hash: str
    ability_specs_hash: str
    slot_rules_hash: str
    legal_action_schema_hash: str
    elixir_rules_hash: str
    mode_map_hash: str
    visual_asset_pack_hash: str
    card_spec_count: int
    contract_versions: Mapping[str, str]
    files: Mapping[str, Mapping[str, str]]
    config: Mapping[str, Any]
    ruleset_id: str = ""
    version: str = field(default=RULESET_MANIFEST_VERSION, init=False)
    VERSION = RULESET_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not self.ruleset_version:
            raise ContractError("ruleset_version must not be empty")
        hash_fields = (
            "engine_build_hash",
            "simulator_build_hash",
            "card_schema_hash",
            "observation_contract_hash",
            "action_contract_hash",
            "tick_rng_latency_rules_hash",
            "card_specs_hash",
            "ability_specs_hash",
            "slot_rules_hash",
            "legal_action_schema_hash",
            "elixir_rules_hash",
            "mode_map_hash",
            "visual_asset_pack_hash",
        )
        for name in hash_fields:
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise ContractError(f"{name} must be a lowercase SHA-256 digest")
        if self.card_spec_count <= 0:
            raise ContractError("card_spec_count must be positive")
        object.__setattr__(self, "contract_versions", frozen_mapping(self.contract_versions))
        frozen_groups = {str(group): frozen_mapping(values) for group, values in self.files.items()}
        for group, values in frozen_groups.items():
            for path, digest in values.items():
                if not path or not _SHA256.fullmatch(str(digest)):
                    raise ContractError(f"invalid file hash in group {group!r}: {path!r}")
        object.__setattr__(self, "files", frozen_mapping(frozen_groups))
        object.__setattr__(self, "config", frozen_mapping(self.config))
        object.__setattr__(self, "ruleset_id", self.calculate_ruleset_id())

    def identity_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("ruleset_id", None)
        return value

    def calculate_ruleset_id(self) -> str:
        return content_hash(self.identity_dict())

    @property
    def patch_hash(self) -> str:
        """Content identity of game/runtime data, excluding Python wrappers."""

        return content_hash(
            {
                "ruleset_version": self.ruleset_version,
                "engine_build_hash": self.engine_build_hash,
                "card_specs_hash": self.card_specs_hash,
                "ability_specs_hash": self.ability_specs_hash,
                "elixir_rules_hash": self.elixir_rules_hash,
                "mode_map_hash": self.mode_map_hash,
                "visual_asset_pack_hash": self.visual_asset_pack_hash,
                "data_files": self.files.get("data", {}),
            }
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RulesetManifestV1":
        if value.get("version") != RULESET_MANIFEST_VERSION:
            raise ContractError(f"unsupported ruleset manifest: {value.get('version')}")
        data = dict(value)
        data.pop("version", None)
        if not data.get("ruleset_id"):
            raise ContractError("ruleset manifest wire object requires ruleset_id")
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists() and destination.is_dir():
            destination = destination / f"{self.ruleset_id}.json"
        elif not destination.suffix:
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / f"{self.ruleset_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json(pretty=True) + "\n"
        if destination.exists():
            existing = destination.read_text(encoding="utf-8")
            if existing != payload:
                raise FileExistsError(f"immutable ruleset path already contains different bytes: {destination}")
            return destination
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RulesetManifestV1":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(value)


def build_ruleset_manifest(
    workspace_root: str | Path | None = None,
    *,
    catalog: CardSpecCatalog | None = None,
    ruleset_version: str | None = None,
    config: Mapping[str, Any] | None = None,
    engine_paths: Iterable[str | Path] = (),
    simulator_paths: Iterable[str | Path] = (),
    data_paths: Iterable[str | Path] = (),
    visual_paths: Iterable[str | Path] = (),
    environment_config: Mapping[str, Any] | EnvironmentConfigV1 | None = None,
    static_card_logic: StaticCardLogicCatalogV1 | None = None,
) -> RulesetManifestV1:
    workspace = Path(workspace_root or WORKSPACE_ROOT).resolve()
    card_catalog = catalog or build_card_catalog(workspace)
    static_card_logic = (
        static_card_logic
        if static_card_logic is not None
        else build_static_card_logic_catalog(workspace, card_catalog=card_catalog)
    )
    native_effect_catalog = build_effect_catalog(static_card_logic, workspace)
    native_projectile_catalog = build_projectile_catalog(static_card_logic, workspace)
    semantic_subset = build_semantic_supported_subset(card_catalog)
    normal_mode_readiness = load_normal_mode_policy_readiness()
    normal_mode_case_scope = getattr(normal_mode_readiness, "normal_mode_case_scope", None)
    runtime_evidence_scope = build_normal_mode_runtime_policy_scope(
        normal_mode_scope_version=semantic_subset.normal_mode_scope_version,
        eligible_card_ids=tuple(sorted(semantic_subset.allowed_ids)),
        normal_mode_case_scope=normal_mode_case_scope,
    )
    standard_mode_timeline = load_game_mode_timeline(72_000_006, workspace)
    effective_ruleset_version = ruleset_version or _read_patch_version(workspace)
    runtime_content_version, runtime_content_manifest_sha1 = _read_runtime_fingerprint(workspace)

    engine = _discover_engine_files(workspace)
    engine.extend(Path(item).resolve() for item in engine_paths)
    simulator = [PACKAGE_ROOT / "probe" / "out" / "libcrprobe.so", *sorted(PACKAGE_ROOT.glob("*.py"))]
    simulator.extend(
        path
        for path in _iter_files(PACKAGE_ROOT / "probe", {".cpp", ".h", ".inc", ".s", ".ps1"})
        if not path.name.startswith("test_")
    )
    simulator.extend(Path(item).resolve() for item in simulator_paths)
    schema = [
        PACKAGE_ROOT / name
        for name in (
            "contracts.py",
            "card_specs.py",
            "normal_form_evidence.py",
            "card_logic.py",
            "effect_catalog.py",
            "projectile_catalog.py",
            "semantic_subset.py",
            "runtime_scope.py",
            "ruleset.py",
        )
    ]
    data = [workspace / name for name in card_catalog.source_files if not Path(name).is_absolute()]
    data.extend(Path(name) for name in card_catalog.source_files if Path(name).is_absolute())
    # ``build_card_catalog`` first discovers the decoded release by requiring
    # both of these plaintext tables, even when a runtime overlay supplies the
    # effective spells_characters rows.  They are therefore runtime closure
    # inputs, not incidental source-tree files.
    catalog_source = workspace / card_catalog.source_release / "assets" / "csv_logic"
    data.extend(
        [
            catalog_source / "spells_characters.csv",
            catalog_source / "spells_other.csv",
            workspace
            / "captures"
            / "decompressed_assets"
            / card_catalog.source_release
            / "assets"
            / "csv_logic"
            / "spells_hero_form.csv",
        ]
    )
    data.extend(_iter_files(workspace / "runtime-update" / "csv_logic", {".csv", ".toml"}))
    data.extend(
        [
            workspace / "runtime-update" / "fingerprint.json",
            workspace / "runtime-update" / "data_manifest.toml",
            PACKAGE_ROOT / "data" / "native_match.json",
            Path(standard_mode_timeline.game_modes_source),
            Path(standard_mode_timeline.timeline.source_path),
        ]
    )
    data.extend(Path(item).resolve() for item in data_paths)
    visual = [workspace / "runtime-update" / "fingerprint.json", workspace / "runtime-update" / "assets.scdb"]
    visual.extend(Path(item).resolve() for item in visual_paths)

    file_groups = {
        "engine": _hash_paths(engine, workspace),
        "simulator": _hash_paths(simulator, workspace),
        "schema": _hash_paths(schema, workspace),
        "data": _hash_paths(data, workspace),
        "visual": _hash_paths(visual, workspace),
    }
    if not file_groups["engine"]:
        raise FileNotFoundError("no ARM64 libg.so found for ruleset manifest")
    if not file_groups["simulator"]:
        raise FileNotFoundError("no native simulator implementation found")
    libg_build_ids = {build_id for path in engine if (build_id := _elf_gnu_build_id(path))}
    if len(libg_build_ids) != 1:
        raise ContractError("ruleset requires exactly one readable ARM64 GNU libg build ID")
    runtime_libg_build_id = next(iter(libg_build_ids))

    descriptor = contract_schema_descriptor()
    contract_versions = dict(descriptor["versions"])
    contract_versions.update({"normal_mode_runtime_policy_scope": (NORMAL_MODE_RUNTIME_POLICY_SCOPE_VERSION)})
    effective_environment = (
        environment_config
        if isinstance(environment_config, EnvironmentConfigV1)
        else EnvironmentConfigV1.from_mapping(environment_config)
    )
    default_config: dict[str, Any] = {
        "tick": {"duration_ms": DEFAULT_TICK_MS, "server_hz": 1000 // DEFAULT_TICK_MS},
        "rng": {"seed_bits": 32, "owner": "native_engine", "deterministic_reset": True},
        "latency": {"contract": LATENCY_CONFIG_VERSION, "applied_inside_battle_env": False},
        "environment": effective_environment.to_dict(),
        "board": {"width": BOARD_WIDTH, "height": BOARD_HEIGHT},
        "slots": {"hand": 4, "deck": 8, "hero_and_wild_are_ruleset_conditioned": True},
        "legal_actions": {
            "kinds": ["wait", "wait_until_affordable", "play_card", "activate_ability"],
            "placement_mask": "card-conditioned-18x32",
            "entity_pointer_targets": False,
            "ability_activation_native_path": {
                "available": True,
                "command": "activate-ability owner object_index secondary_index",
                "source_identity": "stable native entity tuple; private cgid resolved in probe",
                "execution_window": "exact next-tick T/T+1 only",
                "target_arguments": False,
                "fail_closed": True,
            },
        },
        "deck_builder": {
            "distinct_battle_cards": 8,
            "dynamic_card_pointer": True,
            "authoritative_slot_registry": False,
            "missing_rules": [
                "hero_limit",
                "evolution_slots",
                "wild_slots",
                "tower_troop_pool",
                "mode_bans",
                "account_level_slots",
            ],
        },
        "semantic_supported_subset": semantic_subset.binding(),
        "static_card_logic": {
            "version": static_card_logic.version,
            "criteria_version": static_card_logic.criteria_version,
            "catalog_id": static_card_logic.catalog_id,
            "card_count": len(static_card_logic.cards),
            "node_count": len(static_card_logic.nodes),
            "reference_count": len(static_card_logic.references),
        },
        "native_effect_catalog": {
            "version": native_effect_catalog.version,
            "criteria_version": native_effect_catalog.criteria_version,
            "catalog_id": native_effect_catalog.catalog_id,
            "effect_count": len(native_effect_catalog.effects),
        },
        "native_projectile_catalog": {
            "version": native_projectile_catalog.version,
            "criteria_version": native_projectile_catalog.criteria_version,
            "catalog_id": native_projectile_catalog.catalog_id,
            "projectile_count": len(native_projectile_catalog.projectiles),
            "configured_homing_count": native_projectile_catalog.summary["configured_homing_count"],
            "runtime_active_homing_available": native_projectile_catalog.summary["runtime_active_homing_available"],
        },
        "runtime_match_level_policy": {
            "version": RUNTIME_MATCH_LEVEL_POLICY_VERSION,
            "level_cap": RUNTIME_MATCH_LEVEL_CAP,
            "minimum_card_level": RUNTIME_MATCH_MINIMUM_CARD_LEVEL,
            "king_tower_level": RUNTIME_MATCH_KING_TOWER_LEVEL,
        },
        "runtime_evidence_scope": runtime_evidence_scope,
        "elixir": {"max": 10, "actor_exact_own_only": True, "opponent_is_belief": True},
        "engine_source_release": card_catalog.source_release,
        "runtime_content_version": (runtime_content_version or effective_ruleset_version),
        "runtime_content_manifest_sha1": runtime_content_manifest_sha1,
        "runtime_libg_build_id": runtime_libg_build_id,
        "effective_timeline": {
            "game_mode_id": standard_mode_timeline.game_mode_id,
            "game_mode_name": standard_mode_timeline.game_mode_name,
            "timeline_name": standard_mode_timeline.timeline.name,
            "game_modes_source": _workspace_relative_source(standard_mode_timeline.game_modes_source, workspace),
            "game_modes_sha256": standard_mode_timeline.game_modes_sha256,
            "timeline_source": _workspace_relative_source(standard_mode_timeline.timeline.source_path, workspace),
            "timeline_sha256": standard_mode_timeline.timeline.source_sha256,
            "inheritance": (
                "one complete same-root game_modes/battle_timelines pair; "
                "runtime pair preferred with complete capture and decoded-"
                "release fallbacks; SC-compressed source bytes are decoded "
                "in memory while original paths and SHA-256 remain identity-"
                "bearing"
            ),
        },
    }
    if config:
        # Keep caller-specific rules explicit without mutating the defaults.
        default_config["caller"] = dict(config)

    observation_descriptor = {"version": OBSERVATION_VERSION, "fields": descriptor["contracts"]["ObservationV1"]}
    action_descriptor = {"version": ACTION_VERSION, "fields": descriptor["contracts"]["ActionV1"]}
    card_descriptor = {
        "version": descriptor["versions"]["card_spec"],
        "fields": descriptor["contracts"]["CardSpecV1"],
        "static_card_logic_version": static_card_logic.version,
        "static_card_logic_catalog_id": static_card_logic.catalog_id,
        "native_effect_catalog_version": native_effect_catalog.version,
        "native_effect_catalog_id": native_effect_catalog.catalog_id,
        "native_projectile_catalog_version": native_projectile_catalog.version,
        "native_projectile_catalog_id": native_projectile_catalog.catalog_id,
    }
    mode_files = {
        name: digest
        for name, digest in file_groups["data"].items()
        if Path(name).name in {"game_modes.csv", "battle_timelines.csv", "arenas.csv", "arenas_alt.csv"}
    }
    slot_rules = default_config["slots"]
    legal_actions = default_config["legal_actions"]
    elixir_rules = default_config["elixir"]
    tick_rng_latency = {
        "tick": default_config["tick"],
        "rng": default_config["rng"],
        "latency": default_config["latency"],
    }
    return RulesetManifestV1(
        ruleset_version=effective_ruleset_version,
        engine_build_hash=_component_hash(file_groups["engine"]),
        simulator_build_hash=_component_hash(file_groups["simulator"]),
        card_schema_hash=content_hash(card_descriptor),
        observation_contract_hash=content_hash(observation_descriptor),
        action_contract_hash=content_hash(action_descriptor),
        tick_rng_latency_rules_hash=content_hash(tick_rng_latency),
        card_specs_hash=card_catalog.specs_hash,
        ability_specs_hash=card_catalog.ability_specs_hash,
        slot_rules_hash=content_hash(slot_rules),
        legal_action_schema_hash=content_hash(legal_actions),
        elixir_rules_hash=content_hash(elixir_rules),
        mode_map_hash=_component_hash(mode_files, {"fallback": "mode id remains trajectory metadata"}),
        visual_asset_pack_hash=_component_hash(file_groups["visual"]),
        card_spec_count=len(card_catalog.specs),
        contract_versions=contract_versions,
        files=file_groups,
        config=default_config,
    )
