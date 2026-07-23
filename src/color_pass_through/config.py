from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT_ENV = "CPT_DATA_ROOT"


def load_yaml(
    path: str | Path, dataset_root: str | Path | None = None
) -> dict[str, Any]:
    """Load YAML and resolve ``repo://`` and packaged ``dataset://`` paths.

    ``dataset://`` is rooted at the explicit ``dataset_root`` argument or the
    ``CPT_DATA_ROOT`` environment variable.  Requiring a root keeps public
    configs relocatable and prevents silently falling back to private paths.
    """
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    root = dataset_root or os.environ.get(DATASET_ROOT_ENV)
    return resolve_paths(
        data,
        repository_root=REPOSITORY_ROOT,
        dataset_root=Path(root).expanduser().resolve() if root else None,
    )


def resolve_paths(
    value: Any,
    repository_root: Path = REPOSITORY_ROOT,
    dataset_root: Path | None = None,
) -> Any:
    """Recursively resolve portable path URIs without mutating the input."""
    if isinstance(value, Mapping):
        return {
            key: resolve_paths(item, repository_root, dataset_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_paths(item, repository_root, dataset_root) for item in value]
    if isinstance(value, str) and value.startswith("repo://"):
        return str((repository_root / value.removeprefix("repo://")).resolve())
    if isinstance(value, str) and value.startswith("dataset://"):
        if dataset_root is None:
            raise ValueError(
                "Config contains dataset:// paths. Set CPT_DATA_ROOT to the "
                "downloaded color-pass-through dataset root."
            )
        return str((dataset_root / value.removeprefix("dataset://")).resolve())
    return deepcopy(value)


def nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    """Read a required nested value using a dot-separated key."""
    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing config key: {dotted_key}")
        current = current[key]
    return current
