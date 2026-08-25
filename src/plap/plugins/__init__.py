from __future__ import annotations

import tomllib
from importlib.metadata import EntryPoints, entry_points
from pathlib import Path

_MANIFEST = Path(__file__).resolve().parents[3] / "plugins.toml"


def resolve() -> EntryPoints:
    if not _MANIFEST.is_file():
        raise RuntimeError(f"missing plugin manifest: {_MANIFEST}")
    with _MANIFEST.open("rb") as file:
        raw = tomllib.load(file).get("plugins")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("plugins.toml must define a non-empty 'plugins' array")
    if not all(isinstance(name, str) and name for name in raw):
        raise RuntimeError("plugins.toml plugins entries must be non-empty strings")
    if len(raw) != len(set(raw)):
        raise RuntimeError("plugins.toml contains duplicate plugins")
    if "core" not in raw:
        raise RuntimeError("plugins.toml must include 'core'")

    available = entry_points(group="plap.plugin")
    selected = []
    for name in raw:
        matches = [entrypoint for entrypoint in available if entrypoint.name == name]
        if not matches:
            raise RuntimeError(f"configured plugin is not installed: {name!r}")
        if len(matches) > 1:
            raise RuntimeError(f"duplicate configured plugin entrypoint: {name!r}")
        selected.append(matches[0])
    return EntryPoints(selected)


__all__ = ["resolve"]
