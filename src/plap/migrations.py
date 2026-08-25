from __future__ import annotations

import argparse
from importlib.machinery import ModuleSpec, PathFinder
from importlib.metadata import EntryPoint
from importlib.util import find_spec
from pathlib import Path

from alembic import command
from alembic.config import Config

from plap import plugins

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "pyproject.toml"


def _config(database_url: str | None) -> Config:
    config = Config(toml_file=_CONFIG)
    config.set_main_option("version_table_schema", "public")
    if database_url is not None:
        config.set_main_option("sqlalchemy.url", database_url)
    return config


def _module_spec(module: str) -> ModuleSpec | None:
    parts = module.split(".")
    spec = find_spec(parts[0])
    qualified = parts[0]
    for part in parts[1:]:
        if spec is None or spec.submodule_search_locations is None:
            return None
        qualified = f"{qualified}.{part}"
        spec = PathFinder.find_spec(qualified, spec.submodule_search_locations)
    return spec


def _versions(entrypoint: EntryPoint) -> Path | None:
    spec = _module_spec(entrypoint.module)
    if spec is None:
        raise RuntimeError(f"cannot locate plugin module: {entrypoint.module!r}")

    locations = tuple(spec.submodule_search_locations or ())
    if not locations:
        raise RuntimeError(f"plugin entrypoint is not a package: {entrypoint.name!r}")
    if len(locations) != 1:
        raise RuntimeError(f"plugin package has ambiguous locations: {entrypoint.name!r}")

    versions = Path(locations[0]).resolve() / "migrations" / "versions"
    return versions if versions.is_dir() else None


def _configs(database_url: str | None) -> list[tuple[str, Config]]:
    configs = [("plap", _config(database_url))]
    for entrypoint in plugins.resolve():
        versions = _versions(entrypoint)
        if versions is None:
            continue
        config = _config(database_url)
        config.set_main_option("path_separator", "os")
        config.set_main_option("version_locations", str(versions))
        config.set_main_option("version_table", f"alembic_version_{entrypoint.name}")
        config.attributes["target_metadata"] = None
        configs.append((entrypoint.name, config))
    return configs


def upgrade(database_url: str | None = None) -> None:
    for name, config in _configs(database_url):
        config.print_stdout("== %s ==", name)
        command.upgrade(config, "head")


def current(database_url: str | None = None) -> None:
    for name, config in _configs(database_url):
        config.print_stdout("== %s ==", name)
        command.current(config)


def history() -> None:
    for name, config in _configs(None):
        config.print_stdout("== %s ==", name)
        command.history(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("current", "history", "upgrade"))
    args = parser.parse_args()

    if args.command == "upgrade":
        upgrade()
    elif args.command == "current":
        current()
    else:
        history()


if __name__ == "__main__":
    main()


__all__ = ["current", "history", "upgrade"]
