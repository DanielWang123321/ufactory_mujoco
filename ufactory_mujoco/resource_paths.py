"""Read-only paths for repository/editable and wheel installations."""

import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
# Wheels/sdists carry data inside the package. Editable checkouts keep the
# authored XML and configuration paths, including their original hash inputs.
RESOURCE_ROOT = _PACKAGE / "data" if (_PACKAGE / "data").is_dir() else _PACKAGE.parent
CONFIGS_ROOT = RESOURCE_ROOT / "configs"
MODELS_ROOT = Path(os.environ.get("UFACTORY_MUJOCO_MODELS", RESOURCE_ROOT / "models")).expanduser()
