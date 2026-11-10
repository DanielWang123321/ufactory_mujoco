"""Strict YAML recipe validation and checkpoint integrity contracts for RL tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping
import zipfile

import yaml

from .._hashing import canonical_json_sha256, file_sha256
from ..config import load_task_config
from ..resource_paths import CONFIGS_ROOT
from ..robots import ROBOT_PROFILES
from .model_hash import ModelHashError, runtime_model_sha256

REACH_ENV_VERSION = 1
PICK_PLACE_ENV_VERSION = 2
REACH_SUCCESS_DIST_M = 0.005
REACH_ACTION_CONTRACT = "joint_position_delta"
PICK_PLACE_ACTION_CONTRACT = "cartesian_and_g2_gap_increment"
PICK_PLACE_NUM_OBS = 70
PICK_PLACE_NUM_ACTIONS = 4


class ArtifactError(ValueError):
    """Raised when an RL recipe, training config, or checkpoint manifest fails validation."""


_RECIPE_TOP_KEYS = {"schema_version", "task", "environment", "reward", "train"}

_REACH_ENV_RECIPE_KEYS = {
    "env_version",
    "control",
    "max_steps",
    "decimation",
    "delta",
    "success_dist",
    "q0_noise",
    "validate_targets",
}
_REACH_REWARD_RECIPE_KEYS = {
    "k_shape",
    "success_bonus",
    "step_cost",
    "action_cost",
    "limit_cost",
    "limit_margin",
}
_REACH_TRAIN_RECIPE_KEYS = {
    "algorithm",
    "total_steps",
    "robot_total_steps",
    "n_steps",
    "batch_size",
    "n_epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "net_arch",
    "save_interval",
    "eval_episodes",
    "seed",
}

_PICK_PLACE_ENV_RECIPE_KEYS = {
    "env_version",
    "control",
    "max_steps",
    "decimation",
    "action_scale",
    "action_clip",
    "gap_action_scale",
    "setpoint_leash_m",
    "pre_grasp_gap_m",
    "close_floor_gap_m",
    "workspace_lower",
    "workspace_upper",
    "spawn_xy_lower",
    "spawn_xy_upper",
    "g2_speed_percent",
    "g2_force_percent",
    "reward_gamma",
    "place_clearance_m",
    "release_max_height_m",
    "press_limit_m",
    "near_table_m",
    "max_near_table_descent_m_s",
    "table_overload_margin_n",
    "place_success_dist_m",
    "drag_limit_m",
    "tilt_limit_deg",
    "success_hold_steps",
    "num_obs",
    "num_actions",
}
_PICK_PLACE_REWARD_RECIPE_KEYS = {
    "approach_potential",
    "pregrasp_potential",
    "grasp_potential",
    "lift_potential",
    "transit_potential",
    "place_potential",
    "release_potential",
    "hold_potential",
    "first_contact",
    "first_carry",
    "first_lift",
    "first_touchdown",
    "first_place",
    "success",
    "failure_penalty",
    "fast_descent_penalty",
    "press_penalty",
    "recontact_penalty",
    "table_overload_penalty",
    "high_release_penalty",
    "robot_table_penalty",
    "action_penalty",
    "action_saturation_penalty",
    "step_penalty",
}
_PICK_PLACE_TRAIN_RECIPE_KEYS = {
    "algorithm",
    "total_steps",
    "n_steps",
    "batch_size",
    "n_epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_range",
    "vf_coef",
    "ent_coef",
    "max_grad_norm",
    "net_arch",
    "activation",
    "init_std",
    "save_interval",
    "eval_interval",
    "eval_episodes",
    "early_stop_evaluations",
    "min_success_rate",
    "seed",
}

_REACH_RUNTIME_ENV_CONTRACT_KEYS = (
    "task_name",
    "env_version",
    "control",
    "num_obs",
    "num_actions",
    "max_steps",
    "decimation",
    "delta",
    "success_dist",
    "q0_noise",
    "target_low",
    "target_high",
)

_PICK_PLACE_RUNTIME_ENV_CONTRACT_KEYS = (
    "task_name",
    "env_version",
    "control",
    "runtime_config_sha256",
    "runtime_model_sha256",
    "num_obs",
    "num_actions",
    "max_steps",
    "decimation",
    "action_scale",
    "action_clip",
    "gap_action_scale",
    "setpoint_leash_m",
    "pre_grasp_gap_m",
    "close_floor_gap_m",
    "workspace_lower",
    "workspace_upper",
    "spawn_xy_lower",
    "spawn_xy_upper",
    "g2_speed_percent",
    "g2_force_percent",
    "reward_gamma",
    "place_clearance_m",
    "release_max_height_m",
    "press_limit_m",
    "near_table_m",
    "max_near_table_descent_m_s",
    "table_overload_margin_n",
    "place_success_dist_m",
    "drag_limit_m",
    "tilt_limit_deg",
    "success_hold_steps",
    "cube_xy",
    "target_xy",
    "cube_half_m",
    "cube_mass_kg",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ArtifactError(f"configuration contains unsupported type: {type(value).__name__}")


def _finite(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"{path} must be finite")


def runtime_env_contract_keys(env: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the runtime environment contract keys for the given task."""
    if str(env.get("task_name", "")) == "reach":
        return _REACH_RUNTIME_ENV_CONTRACT_KEYS
    return _PICK_PLACE_RUNTIME_ENV_CONTRACT_KEYS


def runtime_env_contract(env: Mapping[str, Any]) -> dict[str, Any]:
    """Return the task fields that must match between saved config and evaluation."""
    return {key: _plain(env.get(key)) for key in runtime_env_contract_keys(env)}


def load_training_recipe(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a declarative RL recipe YAML."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot read training recipe {source}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _RECIPE_TOP_KEYS:
        raise ArtifactError(f"training recipe top-level fields must be exactly {sorted(_RECIPE_TOP_KEYS)}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ArtifactError("recipe schema_version must be 1")
    task = raw["task"]
    if task not in ("reach", "pick_place"):
        raise ArtifactError(f"unknown RL task in recipe: {task!r}")
    for section in ("environment", "reward", "train"):
        if not isinstance(raw[section], dict):
            raise ArtifactError(f"recipe section {section!r} must be a mapping")
    _finite(raw, path="recipe")

    env = raw["environment"]
    reward = raw["reward"]
    train = raw["train"]
    if task == "reach":
        unknown_env = set(env) - _REACH_ENV_RECIPE_KEYS
        if unknown_env:
            raise ArtifactError(f"unsupported reach environment recipe fields: {sorted(unknown_env)}")
        unknown_reward = set(reward) - _REACH_REWARD_RECIPE_KEYS
        if unknown_reward:
            raise ArtifactError(f"unsupported reach reward recipe fields: {sorted(unknown_reward)}")
        unknown_train = set(train) - _REACH_TRAIN_RECIPE_KEYS
        if unknown_train:
            raise ArtifactError(f"unsupported reach train recipe fields: {sorted(unknown_train)}")
        if float(env.get("success_dist", 0.0)) != REACH_SUCCESS_DIST_M:
            raise ArtifactError(f"reach success_dist must be {REACH_SUCCESS_DIST_M} (5 mm required)")
        if str(env.get("control", "")) != REACH_ACTION_CONTRACT:
            raise ArtifactError(f"reach control must be {REACH_ACTION_CONTRACT!r}")
    else:
        runtime_owned = {"cube_xy", "target_xy", "cube_half_m", "cube_mass_kg", "object_mass_kg"} & set(env)
        if runtime_owned:
            raise ArtifactError(
                f"physical task geometry belongs in TaskConfig YAML, not recipe: {sorted(runtime_owned)}"
            )
        unknown_env = set(env) - _PICK_PLACE_ENV_RECIPE_KEYS
        if unknown_env:
            raise ArtifactError(f"unsupported pick-place RL environment fields: {sorted(unknown_env)}")
        unknown_reward = set(reward) - _PICK_PLACE_REWARD_RECIPE_KEYS
        if unknown_reward:
            raise ArtifactError(f"unsupported pick-place RL reward fields: {sorted(unknown_reward)}")
        unknown_train = set(train) - _PICK_PLACE_TRAIN_RECIPE_KEYS
        if unknown_train:
            raise ArtifactError(f"unsupported pick-place RL train fields: {sorted(unknown_train)}")
        if int(env.get("env_version", 0)) != PICK_PLACE_ENV_VERSION:
            raise ArtifactError(f"pick-place env_version must be {PICK_PLACE_ENV_VERSION}")
        if str(env.get("control", "")) != PICK_PLACE_ACTION_CONTRACT:
            raise ArtifactError(f"pick-place control must be {PICK_PLACE_ACTION_CONTRACT!r}")
        if int(env.get("num_obs", 0)) != PICK_PLACE_NUM_OBS or int(env.get("num_actions", 0)) != PICK_PLACE_NUM_ACTIONS:
            raise ArtifactError(
                f"pick-place requires num_obs={PICK_PLACE_NUM_OBS} and num_actions={PICK_PLACE_NUM_ACTIONS}"
            )
        ws_lo = env.get("workspace_lower")
        ws_hi = env.get("workspace_upper")
        if (
            not isinstance(ws_lo, list | tuple)
            or not isinstance(ws_hi, list | tuple)
            or len(ws_lo) != 3
            or len(ws_hi) != 3
            or any(float(ws_lo[i]) >= float(ws_hi[i]) for i in range(3))
        ):
            raise ArtifactError("workspace_lower and workspace_upper must be ordered 3D bounds")
        if "spawn_xy_lower" in env or "spawn_xy_upper" in env:
            sp_lo = env.get("spawn_xy_lower")
            sp_hi = env.get("spawn_xy_upper")
            if (
                not isinstance(sp_lo, list | tuple)
                or not isinstance(sp_hi, list | tuple)
                or len(sp_lo) != 2
                or len(sp_hi) != 2
                or any(float(sp_lo[i]) > float(sp_hi[i]) for i in range(2))
                or any(float(sp_lo[i]) < float(ws_lo[i]) or float(sp_hi[i]) > float(ws_hi[i]) for i in range(2))
            ):
                raise ArtifactError("spawn_xy_lower/upper must be ordered 2D bounds within workspace XY bounds")
        for g2_key in ("g2_speed_percent", "g2_force_percent"):
            if g2_key in env and not (0.0 < float(env[g2_key]) <= 100.0):
                raise ArtifactError(f"{g2_key} must be in (0, 100]")
    return _plain(raw)


def build_reach_task_configs(
    robot: str,
    *,
    recipe_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve reach environment, reward, and training configs for ``robot``."""
    if robot not in ROBOT_PROFILES:
        raise ValueError(f"unknown robot {robot!r}; expected one of {sorted(ROBOT_PROFILES)}")
    recipe = load_training_recipe(recipe_path)
    if recipe["task"] != "reach":
        raise ArtifactError(f"expected reach recipe, got {recipe['task']!r}")
    profile = ROBOT_PROFILES[robot]
    n = profile.n_joints
    env_cfg = deepcopy(recipe["environment"])
    env_cfg.update(
        {
            "task_name": "reach",
            "robot": robot,
            "num_obs": 3 + 3 * n,
            "num_actions": n,
            "target_low": list(profile.target_low),
            "target_high": list(profile.target_high),
            "shoulder": list(profile.shoulder),
            "reach_min": float(profile.reach_min),
            "reach_max": float(profile.reach_max),
        }
    )
    if float(env_cfg["success_dist"]) != REACH_SUCCESS_DIST_M:
        raise ValueError("success_dist must be 0.005 (5 mm required)")
    reward_cfg = deepcopy(recipe["reward"])
    train_cfg = deepcopy(recipe["train"])
    robot_steps = train_cfg.pop("robot_total_steps", {}) or {}
    if robot in robot_steps:
        train_cfg["total_steps"] = int(robot_steps[robot])
    return env_cfg, reward_cfg, train_cfg


def build_pick_place_task_configs(
    robot: str = "xarm6",
    *,
    recipe_path: str | Path,
    config_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve xArm6 + Gripper G2 pick-and-place RL configs bound to TaskConfig SHA-256."""
    if robot != "xarm6":
        raise ValueError(f"pick-place RL only supports xArm6, got {robot!r}")
    recipe = load_training_recipe(recipe_path)
    if recipe["task"] != "pick_place":
        raise ArtifactError(f"expected pick_place recipe, got {recipe['task']!r}")
    overlay = config_path if config_path is not None else (CONFIGS_ROOT / "pick_place" / "xarm6.yaml")
    task_config = load_task_config(overlay)
    if task_config.robot != "xarm6" or task_config.gripper != "gripper_g2" or task_config.scene != "pick_place":
        raise ValueError("pick-place RL requires robot=xarm6, gripper=gripper_g2, scene=pick_place")
    pp = task_config.pick_place
    env_cfg = deepcopy(recipe["environment"])
    env_cfg.setdefault("spawn_xy_lower", [0.30, -0.20])
    env_cfg.setdefault("spawn_xy_upper", [0.50, 0.20])
    env_cfg.setdefault("g2_speed_percent", 50.0)
    env_cfg.setdefault("g2_force_percent", 50.0)
    env_cfg.update(
        {
            "task_name": "pick_place",
            "robot": "xarm6",
            "gripper": "gripper_g2",
            "runtime_config_sha256": task_config.sha256,
            "runtime_model_sha256": runtime_model_sha256(model_path or task_config.scene_path),
            "cube_xy": list(pp.cube_xy),
            "target_xy": list(pp.target_xy),
            "cube_half_m": float(pp.cube_half_m),
            "cube_mass_kg": float(pp.cube_mass_kg),
            "grasp_gap_m": float(pp.grasp_gap_m),
            "lift_z_m": float(pp.lift_z_m),
            "flange_grasp_z_m": float(pp.flange_grasp_z_m),
        }
    )
    reward_cfg = deepcopy(recipe["reward"])
    train_cfg = deepcopy(recipe["train"])
    return env_cfg, reward_cfg, train_cfg


def write_training_config(
    path: str | Path,
    *,
    task: str,
    robot: str,
    env: Mapping[str, Any],
    reward: Mapping[str, Any],
    train: Mapping[str, Any],
) -> str:
    """Write ``config.yaml`` with a canonical ``config_sha256``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": 1,
        "task": str(task),
        "robot": str(robot),
        "env": _plain(env),
        "reward": _plain(reward),
        "train": _plain(train),
    }
    _finite(body)
    body["config_sha256"] = canonical_json_sha256(body)
    target.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return str(body["config_sha256"])


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load and verify a saved ``config.yaml``."""
    source = Path(path)
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot read training config {source}: {exc}") from exc
    required = {"schema_version", "task", "robot", "env", "reward", "train", "config_sha256"}
    if not isinstance(data, dict) or set(data) != required:
        raise ArtifactError(f"training config fields must be exactly {sorted(required)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ArtifactError("unsupported training config schema")
    _finite(data)
    claimed = str(data.pop("config_sha256"))
    actual = canonical_json_sha256(data)
    data["config_sha256"] = claimed
    if claimed != actual:
        raise ArtifactError("training config hash mismatch")
    for section in ("env", "reward", "train"):
        if not isinstance(data[section], dict):
            raise ArtifactError(f"training config {section} must be a mapping")
    return data


@dataclass(frozen=True)
class CheckpointManifest:
    """Hashes and task/space contract stored next to a saved SB3 checkpoint."""
    schema_version: int
    task: str
    robot: str
    dof: int
    observation_dim: int
    action_dim: int
    action_contract: str
    runtime_config_sha256: str | None
    config_sha256: str
    checkpoint_sha256: str
    code_version: str
    checkpoint_file: str


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    training_config: Mapping[str, Any],
    action_contract: str,
    output_path: str | Path | None = None,
) -> Path:
    """Write ``<stem>.checkpoint_manifest.json`` for a saved SB3 ``.zip`` policy."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise ArtifactError(f"checkpoint does not exist: {checkpoint}")
    env = training_config["env"]
    robot = str(training_config["robot"])
    dof = ROBOT_PROFILES[robot].n_joints if robot in ROBOT_PROFILES else int(env["num_actions"])
    runtime_sha = env.get("runtime_config_sha256")
    manifest = CheckpointManifest(
        schema_version=1,
        task=str(training_config["task"]),
        robot=robot,
        dof=int(dof),
        observation_dim=int(env["num_obs"]),
        action_dim=int(env["num_actions"]),
        action_contract=str(action_contract),
        runtime_config_sha256=None if runtime_sha is None else str(runtime_sha),
        config_sha256=str(training_config["config_sha256"]),
        checkpoint_sha256=file_sha256(checkpoint),
        code_version=_code_version(),
        checkpoint_file=checkpoint.name,
    )
    target = Path(output_path) if output_path is not None else checkpoint.with_suffix(".checkpoint_manifest.json")
    target.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    """Load and validate a ``CheckpointManifest`` JSON file."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read checkpoint manifest {source}: {exc}") from exc
    required = set(CheckpointManifest.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != required:
        raise ArtifactError(f"checkpoint manifest fields must be exactly {sorted(required)}")
    manifest = CheckpointManifest(**data)
    if (
        type(manifest.schema_version) is not int
        or manifest.schema_version != 1
        or len(manifest.checkpoint_sha256) != 64
    ):
        raise ArtifactError("invalid checkpoint manifest schema or hash")
    return manifest


def write_run_provenance(
    path: str | Path,
    *,
    training_config: Mapping[str, Any],
    source_paths: list[str | Path],
) -> Path:
    """Write machine, package, Git, and source-file SHA-256 provenance JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_sources: list[dict[str, str]] = []
    for source_path in source_paths:
        source = Path(source_path).resolve()
        if not source.is_file():
            raise ArtifactError(f"provenance source does not exist: {source}")
        resolved_sources.append({"path": str(source), "sha256": file_sha256(source)})

    packages: dict[str, str | None] = {}
    for name in ("mujoco", "numpy", "scipy", "PyYAML", "gymnasium", "stable_baselines3", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    body = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "code_version": _code_version(),
        "training_config_sha256": str(training_config["config_sha256"]),
        "runtime_env_contract_sha256": canonical_json_sha256(runtime_env_contract(training_config["env"])),
        "sources": resolved_sources,
    }
    target.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _verify_sb3_zip_structure(checkpoint: Path) -> None:
    """Verify the SB3 .zip archive contains safe members before PPO.load()."""
    if not zipfile.is_zipfile(checkpoint):
        raise ArtifactError(f"checkpoint is not a valid ZIP archive: {checkpoint}")
    with zipfile.ZipFile(checkpoint, "r") as archive:
        names = set(archive.namelist())
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ArtifactError(f"unsafe path in checkpoint archive: {name}")
        required = {"data", "policy.pth"}
        missing = sorted(required - names)
        if missing:
            raise ArtifactError(f"SB3 checkpoint archive is missing required members: {missing}")


def validate_checkpoint_artifacts(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    expected_task: str | None = None,
    expected_robot: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_runtime_env: Mapping[str, Any] | None = None,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], CheckpointManifest]:
    """Validate config and manifest hashes and task contracts before deserializing a checkpoint."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise ArtifactError(f"checkpoint does not exist: {checkpoint}")
    cfg_file = Path(config_path) if config_path is not None else (checkpoint.parent / "config.yaml")
    man_file = Path(manifest_path) if manifest_path is not None else checkpoint.with_suffix(".checkpoint_manifest.json")
    config = load_training_config(cfg_file)
    manifest = load_checkpoint_manifest(man_file)

    failures: list[str] = []
    if file_sha256(checkpoint) != manifest.checkpoint_sha256:
        failures.append("checkpoint hash")
    if config["config_sha256"] != manifest.config_sha256:
        failures.append("config hash")
    if manifest.checkpoint_file != checkpoint.name:
        failures.append("checkpoint filename")
    if manifest.task != config["task"] or manifest.robot != config["robot"]:
        failures.append("manifest task/robot")
    if expected_task is not None and (manifest.task != expected_task or config["task"] != expected_task):
        failures.append("task")
    if expected_robot is not None and (manifest.robot != expected_robot or config["robot"] != expected_robot):
        failures.append("robot")

    env = config["env"]
    if manifest.runtime_config_sha256 != env.get("runtime_config_sha256"):
        failures.append("manifest runtime config hash")
    if config["task"] == "reach":
        if float(env.get("success_dist", 0.0)) != REACH_SUCCESS_DIST_M:
            failures.append("reach success_dist (5 mm required)")
        if manifest.action_contract != REACH_ACTION_CONTRACT:
            failures.append("reach action contract")
    elif config["task"] == "pick_place":
        if (env.get("task_name") != "pick_place" or config["robot"] != "xarm6"
                or env.get("robot") != "xarm6" or env.get("gripper") != "gripper_g2"
                or manifest.dof != 6 or manifest.observation_dim != PICK_PLACE_NUM_OBS
                or manifest.action_dim != PICK_PLACE_NUM_ACTIONS):
            failures.append("pick-place robot or space contract")
        if int(env.get("env_version", 0)) != PICK_PLACE_ENV_VERSION:
            failures.append("pick-place env_version (retrain the policy)")
        if manifest.action_contract != PICK_PLACE_ACTION_CONTRACT:
            failures.append("pick-place action contract (retrain the policy)")
        saved_model_hash = env.get("runtime_model_sha256")
        if not isinstance(saved_model_hash, str) or len(saved_model_hash) != 64:
            failures.append("missing or invalid runtime model hash (retrain the policy)")
        else:
            scene = model_path or load_task_config(CONFIGS_ROOT / "pick_place" / "xarm6.yaml").scene_path
            try:
                if saved_model_hash != runtime_model_sha256(scene):
                    failures.append("runtime model hash")
            except ModelHashError as exc:
                raise ArtifactError(str(exc)) from exc

    if expected_runtime_config_sha256 is not None:
        if env.get("runtime_config_sha256") != str(expected_runtime_config_sha256) or (
            manifest.runtime_config_sha256 != str(expected_runtime_config_sha256)
        ):
            failures.append("runtime config hash")

    if expected_runtime_env is not None:
        if env.get("env_version") != expected_runtime_env.get("env_version"):
            raise ArtifactError("environment version changed; retrain the policy")
        saved_keys = runtime_env_contract_keys(env)
        expected_keys = runtime_env_contract_keys(expected_runtime_env)
        if saved_keys != expected_keys:
            failures.append("runtime config body (task_name)")
        else:
            saved_contract = runtime_env_contract(env)
            expected_contract = runtime_env_contract(expected_runtime_env)
            mismatched = [k for k in saved_keys if saved_contract.get(k) != expected_contract.get(k)]
            if mismatched:
                failures.append(f"runtime config body ({', '.join(mismatched)})")

    if (
        int(env.get("num_obs", -1)) != manifest.observation_dim
        or int(env.get("num_actions", -1)) != manifest.action_dim
    ):
        failures.append("observation/action dimensions")

    if failures:
        raise ArtifactError(f"checkpoint artifact mismatch: {', '.join(failures)}")
    return config, manifest


def load_validated_ppo(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    device: str = "cpu",
    expected_task: str | None = None,
    expected_robot: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_runtime_env: Mapping[str, Any] | None = None,
    model_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any], CheckpointManifest]:
    """Validate checkpoint manifest and config hashes before calling SB3 ``PPO.load``."""
    checkpoint = Path(checkpoint_path)
    config, manifest = validate_checkpoint_artifacts(
        checkpoint,
        config_path,
        manifest_path,
        expected_task=expected_task,
        expected_robot=expected_robot,
        expected_runtime_config_sha256=expected_runtime_config_sha256,
        expected_runtime_env=expected_runtime_env,
        model_path=model_path,
    )
    _verify_sb3_zip_structure(checkpoint)
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device=device)
    obs_shape = getattr(model.observation_space, "shape", None)
    act_shape = getattr(model.action_space, "shape", None)
    if obs_shape != (manifest.observation_dim,) or act_shape != (manifest.action_dim,):
        raise ArtifactError(
            f"loaded policy space shapes {obs_shape}/{act_shape} do not match manifest "
            f"({manifest.observation_dim},)/({manifest.action_dim},)"
        )
    return model, config, manifest

