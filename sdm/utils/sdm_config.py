"""Resolve hand-maintained SDM YAML configs from the git checkout."""

from __future__ import annotations

import logging
from pathlib import Path

from pyhere import here

from sdm.utils.fs import filesystem_for_uri, join_uri_path, uri_exists

logger = logging.getLogger(__name__)


def resolve_sdm_config_root(config_or_cli_path: str | Path) -> str:
    """Resolve a repo-managed config path without ``INPUT_DATA_ROOT`` rewriting.

    Applies to the shared ``model_config.yml``, ``variables_config.yml``, and
    ``data/sdm_config/`` trees. Absolute paths and ``gs://`` URIs are returned
    unchanged. Relative paths resolve against the project root (git checkout),
    not cloud input data.
    """
    path_str = str(config_or_cli_path)
    if "://" in path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((Path(here(".")) / path).resolve())


def _basename_from_listing_entry(entry: str) -> str:
    return entry.rstrip("/").split("/")[-1]


def find_subdirectory_case_insensitive(base_dir: str, name: str) -> str | None:
    """Return the path to a subdirectory of ``base_dir`` matching ``name`` case-insensitively."""
    ref = str(base_dir).rstrip("/")
    target_lower = name.lower()
    if "://" in ref:
        fs, path = filesystem_for_uri(ref)
        if not fs.isdir(path):
            return None
        for entry in fs.ls(path, detail=False):
            basename = _basename_from_listing_entry(str(entry))
            if basename.lower() == target_lower:
                return join_uri_path(ref, basename)
        return None

    base = Path(ref)
    if not base.is_dir():
        return None
    for child in base.iterdir():
        if child.is_dir() and child.name.lower() == target_lower:
            return str(child.resolve())
    return None


def resolve_model_config_dir(
    config_root: str | Path,
    model_id: str,
    *,
    must_exist: bool = True,
) -> str:
    """Resolve the per-model config folder under ``config_root``.

    When ``must_exist`` is true, a missing folder raises ``FileNotFoundError``.
    When false, an existing folder is found case-insensitively; otherwise the path
    ``{config_root}/{model_id}`` is returned for creation.
    """
    root = resolve_sdm_config_root(config_root)
    if not uri_exists(root):
        raise FileNotFoundError(
            f"SDM config directory not found at {root}. "
            f"Expected hand-maintained per-model configs under this path."
        )

    matched = find_subdirectory_case_insensitive(root, model_id)
    if matched is not None:
        return matched

    expected = join_uri_path(root, model_id)
    if must_exist:
        raise FileNotFoundError(
            f"No per-model SDM config folder for model id '{model_id}'. "
            f"Expected a subdirectory of {root} matching '{model_id}' "
            f"(case-insensitive), e.g. {expected}."
        )
    return expected


def model_config_file_paths(config_dir: str) -> tuple[str, str]:
    """Return ``(model_config.yml, variables_config.yml)`` paths under ``config_dir``."""
    model_path = join_uri_path(config_dir, "model_config.yml")
    variables_path = join_uri_path(config_dir, "variables_config.yml")
    return model_path, variables_path


def repo_relative_config_path(config_or_cli_path: str | Path) -> str:
    """Return a config path relative to the repo root when it lives in the checkout."""
    path_str = str(config_or_cli_path)
    if "://" in path_str:
        return path_str
    path = Path(path_str)
    repo_root = Path(here(".")).resolve()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (repo_root / path).resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return path_str
