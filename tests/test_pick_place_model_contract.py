"""Model fingerprint and pick-place checkpoint contracts."""

import json
from pathlib import Path
import shutil
import unittest
import zipfile

from scripts.train_pick_place_rl import write_pick_place_artifacts
from ufactory_mujoco.rl import (
    ArtifactError, PICK_PLACE_ACTION_CONTRACT, REACH_ACTION_CONTRACT,
    build_pick_place_task_configs, build_reach_task_configs, load_training_config,
    validate_checkpoint_artifacts, write_checkpoint_manifest, write_training_config,
)
from ufactory_mujoco.rl.model_hash import ModelHashError, runtime_model_sha256

try:
    import pytest
except ImportError as exc:
    raise unittest.SkipTest("pytest is required for test_pick_place_model_contract") from exc

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "examples/rl/pick_place/recipe.yaml"


def _tiny_scene(path: Path):
    path.mkdir()
    (path / "scene.xml").write_text('<mujoco><include file="a.xml"/></mujoco>')
    (path / "a.xml").write_text('<mujoco><compiler meshdir="assets"/><include file="b.xml"/></mujoco>')
    (path / "b.xml").write_text('<mujoco><asset><mesh name="part" file="part.stl"/></asset></mujoco>')
    (path / "assets").mkdir()
    (path / "assets/part.stl").write_bytes(b"first mesh")
    return path / "scene.xml"


def _checkpoint(path, env, reward, train, *, task="pick_place", contract=PICK_PLACE_ACTION_CONTRACT):
    path.mkdir(parents=True)
    checkpoint = path / "ppo.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", "{}")
        archive.writestr("policy.pth", "fixture")
    write_training_config(path / "config.yaml", task=task, robot="xarm6", env=env, reward=reward, train=train)
    config = load_training_config(path / "config.yaml")
    write_checkpoint_manifest(checkpoint, training_config=config, action_contract=contract)
    (path / "provenance.json").write_text(json.dumps({"training_config_sha256": config["config_sha256"]}))
    return checkpoint


def test_model_hash_recursive_dependencies_and_relocation(tmp_path):
    scene = _tiny_scene(tmp_path / "source")
    original = runtime_model_sha256(scene)
    staged = tmp_path / "installed"
    shutil.copytree(scene.parent, staged)
    (staged / "assets").rename(staged / "meshes")
    xml = staged / "a.xml"
    xml.write_text(xml.read_text().replace('meshdir="assets"', 'meshdir="meshes"'))
    assert runtime_model_sha256(staged / "scene.xml") == original
    (staged / "meshes/part.stl").write_bytes(b"different mesh")
    assert runtime_model_sha256(staged / "scene.xml") != original
    (staged / "meshes/part.stl").unlink()
    with pytest.raises(ModelHashError, match="cannot read model dependency"):
        runtime_model_sha256(staged / "scene.xml")
    (scene.parent / "b.xml").write_text('<mujoco><asset><mesh name="other" file="part.stl"/></asset></mujoco>')
    assert runtime_model_sha256(scene) != original
    (scene.parent / "b.xml").write_text('<mujoco><include file="missing.xml"/></mujoco>')
    with pytest.raises(ModelHashError, match="cannot read model XML"):
        runtime_model_sha256(scene)


def test_v2_checkpoint_rejects_changed_or_missing_model_and_v1(tmp_path):
    scene = _tiny_scene(tmp_path / "model")
    env, reward, train = build_pick_place_task_configs(recipe_path=RECIPE, model_path=scene)
    checkpoint = _checkpoint(tmp_path / "v2", env, reward, train)
    validate_checkpoint_artifacts(checkpoint, model_path=scene, expected_runtime_env=env)
    (scene.parent / "assets/part.stl").write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="runtime model hash"):
        validate_checkpoint_artifacts(checkpoint, model_path=scene)
    (scene.parent / "assets/part.stl").unlink()
    with pytest.raises(ArtifactError, match="cannot read model dependency"):
        validate_checkpoint_artifacts(checkpoint, model_path=scene)
    old_env = dict(env, env_version=1)
    old_env.pop("runtime_model_sha256")
    old = _checkpoint(tmp_path / "v1", old_env, reward, train)
    with pytest.raises(ArtifactError, match="env_version|model hash"):
        validate_checkpoint_artifacts(old, model_path=scene)
    with pytest.raises(ArtifactError, match="environment version changed; retrain the policy"):
        validate_checkpoint_artifacts(old, model_path=scene, expected_runtime_env=env)
    missing_hash = dict(env)
    missing_hash.pop("runtime_model_sha256")
    checkpoint_without_hash = _checkpoint(tmp_path / "missing_hash", missing_hash, reward, train)
    with pytest.raises(ArtifactError, match="missing or invalid runtime model hash"):
        validate_checkpoint_artifacts(checkpoint_without_hash, model_path=scene)


def test_training_refuses_to_rebind_a_changed_model(tmp_path):
    env, reward, _ = build_pick_place_task_configs(recipe_path=RECIPE)
    with pytest.raises(ValueError, match="runtime model changed"):
        write_pick_place_artifacts(tmp_path, robot="xarm6", recipe_path=RECIPE,
                                   expected_model_sha256="0" * 64,
                                   expected_env=env, expected_reward=reward)
    assert not (tmp_path / "config.yaml").exists()


def test_reach_checkpoint_does_not_require_pick_place_model_hash(tmp_path):
    env, reward, train = build_reach_task_configs("xarm6", recipe_path=ROOT / "examples/rl/reach/recipe.yaml")
    checkpoint = _checkpoint(tmp_path / "reach", env, reward, train, task="reach", contract=REACH_ACTION_CONTRACT)
    validate_checkpoint_artifacts(checkpoint, expected_task="reach", expected_runtime_env=env)
