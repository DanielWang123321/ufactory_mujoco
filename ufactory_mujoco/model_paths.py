"""Resource roots and model path resolution for editable and wheel installs."""

import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
# Wheels/sdists carry data inside the package. Editable checkouts keep the
# authored XML and configuration paths, including their original hash inputs.
RESOURCE_ROOT = _PACKAGE / "data" if (_PACKAGE / "data").is_dir() else _PACKAGE.parent
CONFIGS_ROOT = RESOURCE_ROOT / "configs"
MODELS_ROOT = Path(os.environ.get("UFACTORY_MUJOCO_MODELS", RESOURCE_ROOT / "models")).expanduser()


def resolve_model_path(model_arg: str) -> Path:
    """Resolve a model name ('xarm6') or XML path to a loadable scene/model file."""
    p = Path(model_arg)
    if p.is_file():
        return p
    candidates = [
        MODELS_ROOT / model_arg / "scene.xml",
        MODELS_ROOT / model_arg / f"{model_arg}.xml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    available = (
        sorted(d.name for d in MODELS_ROOT.iterdir() if (d / "scene.xml").is_file())
        if MODELS_ROOT.is_dir()
        else []
    )
    raise FileNotFoundError(
        f"Model '{model_arg}' not found. Searched:\n"
        f"  - {p}\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n"
        f"Available models with scene.xml: {available}"
    )
