"""SHA-256 helpers for files and canonical JSON.

These digests are stored in task configs and RL checkpoint manifests; changing
the encoding invalidates every saved artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(data: Any) -> str:
    """Hex SHA-256 of ``data`` as sorted-key, compact, ASCII-escaped JSON."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
