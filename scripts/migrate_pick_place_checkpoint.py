#!/usr/bin/env python3
"""Evaluate a verified v1 pick-place checkpoint before binding it to the v2 model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.rl.pick_place.evaluate import (  # noqa: E402
    DEFAULT_RECIPE, acceptance_passed, evaluate_policy,
)
from scripts.train_pick_place_rl import _env_kwargs  # noqa: E402
from ufactory_mujoco.rl import (  # noqa: E402
    PICK_PLACE_ACTION_CONTRACT, PICK_PLACE_ENV_VERSION, PickPlaceEnv,
    build_pick_place_task_configs, load_training_config, validate_checkpoint_artifacts,
    write_checkpoint_manifest, write_training_config,
)
from ufactory_mujoco.rl.artifacts import (  # noqa: E402
    ArtifactError, _validate_checkpoint_artifacts, _verify_sb3_zip_structure,
)
from ufactory_mujoco.rl.model_hash import verify_runtime_model  # noqa: E402

DEFAULT_OUTPUT = ROOT / "build" / "rl" / "pick_place" / "xarm6_g2_v2"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(checkpoint: Path, output: Path, *, recipe: Path = DEFAULT_RECIPE,
            runtime_config: Path | None = None, seed: int = 17000, device: str = "cpu") -> bool:
    """Create a v2 record only if 20 new-model episodes meet all acceptance limits."""
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if output == checkpoint.parent or (output.exists() and any(output.iterdir())):
        raise ArtifactError(f"migration output must be a new or empty directory: {output}")

    current_env, _, _ = build_pick_place_task_configs(
        "xarm6", recipe_path=recipe, config_path=runtime_config,
    )
    config, manifest = _validate_checkpoint_artifacts(
        checkpoint, expected_task="pick_place", expected_robot="xarm6",
        expected_runtime_config_sha256=current_env["runtime_config_sha256"],
        expected_runtime_env=current_env, legacy_migration=True,
    )
    legacy_files = [checkpoint, checkpoint.parent / "config.yaml",
                    checkpoint.with_suffix(".checkpoint_manifest.json"), checkpoint.parent / "provenance.json"]
    if not legacy_files[-1].is_file():
        raise ArtifactError("legacy provenance.json is required to preserve the training source")
    try:
        provenance = json.loads(legacy_files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read legacy provenance: {exc}") from exc
    if provenance.get("training_config_sha256") != config["config_sha256"]:
        raise ArtifactError("legacy provenance does not match the training config")
    source_hashes = {path.name: _digest(path) for path in legacy_files}
    _verify_sb3_zip_structure(checkpoint)
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device=device)
    if (getattr(model.observation_space, "shape", None) != (manifest.observation_dim,)
            or getattr(model.action_space, "shape", None) != (manifest.action_dim,)):
        raise ArtifactError("legacy policy spaces do not match the manifest")

    env = PickPlaceEnv(**_env_kwargs("xarm6", runtime_config, current_env, config["reward"]))
    try:
        summary, rows = evaluate_policy(
            env, lambda obs: model.predict(obs, deterministic=True)[0], episodes=20, seed=seed,
        )
        verify_runtime_model(env.scene_path, current_env["runtime_model_sha256"])
    finally:
        env.close()
    if any(_digest(path) != source_hashes[path.name] for path in legacy_files):
        raise ArtifactError("legacy files changed during migration")
    passed = summary.get("episodes") == 20 and len(rows) == 20 and acceptance_passed(
        summary, min_success_rate=0.90, max_press_mm=1.0, max_target_error_mm=20.0,
    )
    report = {"passed": passed, "source_checkpoint_sha256": manifest.checkpoint_sha256,
              "source_files_sha256": source_hashes, "runtime_model_sha256": current_env["runtime_model_sha256"],
              "thresholds": {"episodes": 20, "min_success_rate": 0.90,
                             "max_press_mm": 1.0, "max_mean_target_error_mm": 20.0},
              "summary": summary, "episodes_detail": rows}
    if not passed:
        output.mkdir(parents=True, exist_ok=True)
        (output / "migration_eval.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".g2-migration-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "migration_eval.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        new_env = deepcopy(config["env"])
        new_env.update(env_version=PICK_PLACE_ENV_VERSION, runtime_model_sha256=current_env["runtime_model_sha256"])
        write_training_config(staging / "config.yaml", task="pick_place", robot="xarm6",
                              env=new_env, reward=config["reward"], train=config["train"])
        new_config = load_training_config(staging / "config.yaml")
        destination = staging / "ppo_pick_place.zip"
        shutil.copyfile(checkpoint, destination)
        write_checkpoint_manifest(destination, training_config=new_config, action_contract=PICK_PLACE_ACTION_CONTRACT)
        source_dir = staging / "legacy_source"
        source_dir.mkdir()
        for path in legacy_files[1:]:
            shutil.copyfile(path, source_dir / path.name)
        (staging / "provenance.json").write_text(json.dumps({
            "schema_version": 1, "kind": "evaluated_legacy_migration",
            "training_config_sha256": new_config["config_sha256"],
            "original_training_config_sha256": config["config_sha256"],
            "original_training_provenance": "legacy_source/provenance.json",
            "original_files_sha256": source_hashes, "evaluation": "migration_eval.json",
        }, indent=2, sort_keys=True) + "\n")
        validate_checkpoint_artifacts(destination, expected_task="pick_place", expected_robot="xarm6",
                                      expected_runtime_config_sha256=current_env["runtime_config_sha256"],
                                      expected_runtime_env=current_env)
        if any(_digest(path) != source_hashes[path.name] for path in legacy_files):
            raise ArtifactError("legacy files changed during migration")
        if output.exists():
            output.rmdir()  # Only an empty directory may be replaced.
        os.replace(staging, output)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="v1 checkpoint with config, manifest and provenance")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    passed = migrate(args.checkpoint, args.out_dir, recipe=args.recipe,
                     runtime_config=args.runtime_config, seed=args.seed, device=args.device)
    print(f"Migration {'passed' if passed else 'failed'}; report: {args.out_dir / 'migration_eval.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
