"""Setuptools resource staging. Only release/build directories are changed."""

from pathlib import Path
import re
import shutil

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
PACKAGE_DATA = Path("ufactory_mujoco") / "data"


def stage_resources(destination):
    """Copy canonical meshes once and rewrite only staged compiler meshdirs."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    bundled = ROOT / PACKAGE_DATA
    if bundled.is_dir():
        # Rebuilding the wheel from an sdist uses the already staged data.
        shutil.copytree(bundled, destination)
        return
    for source in sorted((ROOT / "models").glob("*/*.xml")):
        target = destination / "models" / source.relative_to(ROOT / "models")
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        text = re.sub(r'(\bmeshdir\s*=\s*")[^"]*(")', r'\g<1>../../meshes\2', text)
        target.write_text(text, encoding="utf-8")
    for asset_dir in sorted((ROOT / "models").glob("*/assets/*")):
        if asset_dir.is_symlink():
            continue
        for source in sorted(asset_dir.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in {".obj", ".stl", ".mtl", ".png"}:
                continue
            target = destination / "meshes" / asset_dir.name / source.relative_to(asset_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError(f"Duplicate canonical asset: {target}")
            shutil.copyfile(source, target)
    for source in sorted((ROOT / "configs").rglob("*.yaml")):
        target = destination / "configs" / source.relative_to(ROOT / "configs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copytree(ROOT / "licenses", destination / "licenses")
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / name, destination / "licenses" / name)
    if not (destination / "models" / "xarm6" / "scene.xml").is_file():
        raise RuntimeError("Missing model resources")


class BuildPy(build_py):
    def run(self):
        super().run()
        if not self.editable_mode:
            stage_resources(Path(self.build_lib) / PACKAGE_DATA)


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        stage_resources(Path(base_dir) / PACKAGE_DATA)
