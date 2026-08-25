from __future__ import annotations

import sys
from importlib import invalidate_caches
from importlib.metadata import EntryPoint, EntryPoints

import anyio
import pytest
from sqlalchemy import text

from plap import migrations, plugins


def _write_revision(versions, revision: str, down_revision: str | None, body: str) -> None:
    (versions / f"{revision}.py").write_text(
        f'"""Test plugin revision."""\n\nfrom alembic import op\nimport sqlalchemy as sa\n\nrevision = "{revision}"\n'
        f"down_revision = {down_revision!r}\nbranch_labels = None\ndepends_on = None\n\n"
        f"def upgrade():\n{body}\n\ndef downgrade():\n    pass\n"
    )


async def _reset_database(db_session_maker) -> None:
    async with db_session_maker() as session, session.begin():
        await session.execute(text("drop schema if exists migration_test cascade"))
        await session.execute(text("drop schema if exists identity cascade"))
        await session.execute(text("drop schema if exists responses cascade"))
        await session.execute(text("drop schema if exists public cascade"))
        await session.execute(text("create schema public"))


async def _state(db_session_maker) -> tuple[str, str, bool, set[str]]:
    async with db_session_maker() as session:
        version = (await session.execute(text("select version_num from public.alembic_version_migration_test"))).scalar_one()
        second_version = (await session.execute(text("select version_num from public.alembic_version_migration_second"))).scalar_one()
        has_name = (
            await session.execute(
                text(
                    """
                    select exists (
                      select 1
                        from information_schema.columns
                       where table_schema = 'migration_test'
                         and table_name = 'items'
                         and column_name = 'name'
                    )
                    """
                )
            )
        ).scalar_one()
        version_table_schemas = set(
            (
                await session.execute(
                    text(
                        """
                        select table_schema
                          from information_schema.tables
                         where table_name in (
                           'alembic_version',
                           'alembic_version_migration_test',
                           'alembic_version_migration_second'
                         )
                        """
                    )
                )
            ).scalars()
        )
    return str(version), str(second_version), bool(has_name), version_table_schemas


async def _drop_plugin_schema(db_session_maker) -> None:
    async with db_session_maker() as session, session.begin():
        await session.execute(text("drop schema if exists migration_test cascade"))


async def test_plugin_migrations_pause_while_disabled_and_resume_when_enabled(
    db_session_maker,
    monkeypatch: pytest.MonkeyPatch,
    test_config,
    tmp_path,
) -> None:
    package = tmp_path / "migration_test_plugin"
    package.mkdir()
    (package / "__init__.py").write_text('raise RuntimeError("plugin runtime was imported")\n')
    versions = package / "migrations" / "versions"
    versions.mkdir(parents=True)
    _write_revision(
        versions,
        "plugin_0001",
        None,
        "    op.execute('select version_num from public.alembic_version')\n"
        "    op.execute('create schema migration_test')\n"
        "    op.create_table('items', sa.Column('id', sa.Integer(), primary_key=True), schema='migration_test')",
    )
    second_package = tmp_path / "migration_second_plugin"
    second_package.mkdir()
    (second_package / "__init__.py").write_text('raise RuntimeError("second plugin runtime was imported")\n')
    second_versions = second_package / "migrations" / "versions"
    second_versions.mkdir(parents=True)
    _write_revision(
        second_versions,
        "second_0001",
        None,
        "    op.execute('select version_num from public.alembic_version')",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    invalidate_caches()
    entrypoint = EntryPoint(name="migration_test", value="migration_test_plugin", group="plap.plugin")
    second_entrypoint = EntryPoint(name="migration_second", value="migration_second_plugin", group="plap.plugin")
    monkeypatch.setattr(plugins, "resolve", lambda: EntryPoints((entrypoint, second_entrypoint)))

    await _reset_database(db_session_maker)
    try:
        await anyio.to_thread.run_sync(migrations.upgrade, test_config.database_url)
        assert await _state(db_session_maker) == ("plugin_0001", "second_0001", False, {"public"})

        _write_revision(
            versions,
            "plugin_0002",
            "plugin_0001",
            "    op.add_column('items', sa.Column('name', sa.Text()), schema='migration_test')",
        )
        invalidate_caches()
        monkeypatch.setattr(plugins, "resolve", lambda: EntryPoints((second_entrypoint,)))

        await anyio.to_thread.run_sync(migrations.upgrade, test_config.database_url)
        assert await _state(db_session_maker) == ("plugin_0001", "second_0001", False, {"public"})

        monkeypatch.setattr(plugins, "resolve", lambda: EntryPoints((entrypoint, second_entrypoint)))
        await anyio.to_thread.run_sync(migrations.upgrade, test_config.database_url)

        assert await _state(db_session_maker) == ("plugin_0002", "second_0001", True, {"public"})
        assert "migration_test_plugin" not in sys.modules
        assert "migration_second_plugin" not in sys.modules
    finally:
        await _drop_plugin_schema(db_session_maker)
