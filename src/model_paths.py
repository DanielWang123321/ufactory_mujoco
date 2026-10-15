"""Model path resolution shared by examples, scripts and tests.

Single source for locating robot model XMLs under ``models/``; keeps examples
independent of each other (no cross-example imports).
"""

from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


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
    available = [d.name for d in MODELS_ROOT.iterdir() if (d / "scene.xml").is_file()]
    raise FileNotFoundError(
        f"Model '{model_arg}' not found. Searched:\n"
        f"  - {p}\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n"
        f"Available models with scene.xml: {available}"
    )
