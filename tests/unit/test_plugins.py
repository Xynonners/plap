from __future__ import annotations

from importlib.metadata import EntryPoint, EntryPoints

import pytest

from plap import plugins


def _entrypoints(*names: str) -> EntryPoints:
    return EntryPoints(EntryPoint(name=name, value=f"test_plugins.{name}", group="plap.plugin") for name in names)


def test_resolve_preserves_manifest_order(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifest = tmp_path / "plugins.toml"
    manifest.write_text('plugins = ["core", "second", "first"]\n')
    monkeypatch.setattr(plugins, "_MANIFEST", manifest)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: _entrypoints("first", "core", "second"))

    resolved = plugins.resolve()

    assert isinstance(resolved, EntryPoints)
    assert [entrypoint.name for entrypoint in resolved] == ["core", "second", "first"]


def test_resolve_rejects_missing_configured_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifest = tmp_path / "plugins.toml"
    manifest.write_text('plugins = ["core", "missing"]\n')
    monkeypatch.setattr(plugins, "_MANIFEST", manifest)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: _entrypoints("core"))

    with pytest.raises(RuntimeError, match="configured plugin is not installed: 'missing'"):
        plugins.resolve()


def test_resolve_ignores_duplicate_disabled_entrypoints(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifest = tmp_path / "plugins.toml"
    manifest.write_text('plugins = ["core"]\n')
    monkeypatch.setattr(plugins, "_MANIFEST", manifest)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: _entrypoints("disabled", "core", "disabled"))

    resolved = plugins.resolve()

    assert [entrypoint.name for entrypoint in resolved] == ["core"]
