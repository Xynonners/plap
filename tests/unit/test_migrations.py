from __future__ import annotations

import sys
from importlib import invalidate_caches
from importlib.metadata import EntryPoint, EntryPoints

import pytest
from alembic.script import ScriptDirectory

from plap import migrations, plugins


def _write_revision(versions, revision: str) -> None:
    versions.mkdir(parents=True)
    (versions / f"{revision}.py").write_text(
        f'"""Test revision."""\n\nrevision = "{revision}"\ndown_revision = None\nbranch_labels = None\ndepends_on = None\n'
        "\ndef upgrade():\n    pass\n\ndef downgrade():\n    pass\n"
    )


def test_configs_use_plugin_package_without_importing_it(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    parent = tmp_path / "migration_test_parent"
    parent.mkdir()
    (parent / "__init__.py").write_text('raise RuntimeError("plugin parent runtime was imported")\n')
    package = parent / "plugin"
    package.mkdir()
    (package / "__init__.py").write_text('raise RuntimeError("plugin runtime was imported")\n')
    versions = package / "migrations" / "versions"
    _write_revision(versions, "plugin_0001")

    monkeypatch.syspath_prepend(str(tmp_path))
    invalidate_caches()
    entrypoint = EntryPoint(name="migration_test", value="migration_test_parent.plugin", group="plap.plugin")
    monkeypatch.setattr(plugins, "resolve", lambda: EntryPoints((entrypoint,)))

    configs = migrations._configs(None)

    assert [name for name, _ in configs] == ["plap", "migration_test"]
    plugin_config = configs[1][1]
    assert plugin_config.get_main_option("version_table") == "alembic_version_migration_test"
    assert plugin_config.get_main_option("version_table_schema") == "public"
    assert plugin_config.get_version_locations_list() == [str(versions)]
    assert ScriptDirectory.from_config(plugin_config).get_heads() == ["plugin_0001"]
    assert "migration_test_parent" not in sys.modules
    assert "migration_test_parent.plugin" not in sys.modules
