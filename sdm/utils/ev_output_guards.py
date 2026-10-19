"""Refuse EV rebuild writes into live / bucket input locations.

Protected destinations (any match fails validation):

* Local or remote paths whose resolved path has an ``inputs`` path segment
  (e.g. ``inputs/``, ``./inputs/data/evs``, ``/mirror/inputs/evs``).
* Cloud URIs under a bucket ``inputs`` prefix
  (e.g. ``gs://habitat-suitability-modelling-data/inputs`` and anything
  beneath it, including ``…/inputs/data/evs``).
* The current model EV layer directory used by training: ``data/evs``
  (CLI default) and, when configured, the parent directory of
  ``paths.ev_tiff`` / ``paths.evs`` when those resolve under ``data/evs``.
* Any path under ``INPUT_DATA_ROOT`` when that env var is set (local mirror
  or ``gs://…/inputs``), because that root is the live published input set.
"""

from __future__ import annotations

import os
from pathlib import Path

from sdm.utils.model_runs import INPUT_DATA_ROOT_ENV

# Repo-relative directory that ``sdm data`` historically wrote and that
# ``paths.ev_tiff`` / train still treat as the current EV stack location.
DEFAULT_CURRENT_EVS_DIR = Path("data/evs")


class ForbiddenEvOutputError(ValueError):
    """Raised when an EV rebuild output path would overwrite live inputs."""


def _looks_like_uri(path_str: str) -> bool:
    return "://" in path_str


def _uri_has_inputs_segment(uri: str) -> bool:
    """True for ``scheme://host/…/inputs`` or ``…/inputs/…`` (not ``*inputs*`` filenames)."""
    without_scheme = uri.split("://", 1)[-1]
    # Drop query/fragment if present
    without_scheme = without_scheme.split("?", 1)[0].split("#", 1)[0]
    parts = [p for p in without_scheme.split("/") if p]
    return "inputs" in parts


def _path_has_inputs_segment(path: Path) -> bool:
    return "inputs" in path.parts


def _is_under_or_equal(candidate: Path, protected: Path) -> bool:
    try:
        candidate.resolve().relative_to(protected.resolve())
        return True
    except ValueError:
        return False


def current_evs_dirs(
    *,
    cwd: Path | None = None,
    config_evs: str | Path | None = None,
    config_ev_tiff: str | Path | None = None,
) -> list[Path]:
    """Local directories treated as the live EV layer set."""
    base = (cwd or Path.cwd()).resolve()
    dirs: list[Path] = [(base / DEFAULT_CURRENT_EVS_DIR).resolve()]

    for raw in (config_evs, config_ev_tiff):
        if raw is None:
            continue
        text = str(raw)
        if _looks_like_uri(text):
            continue
        path = Path(text)
        if not path.is_absolute():
            path = base / path
        # For a file (ev_tiff), protect its parent; for a directory, protect it.
        target = path if path.suffix == "" else path.parent
        # Only treat as "current EV set" when under data/evs or named evs.
        resolved = target.resolve()
        if resolved == dirs[0] or _is_under_or_equal(resolved, dirs[0]):
            continue
        if resolved.name == "evs" or DEFAULT_CURRENT_EVS_DIR.as_posix() in resolved.as_posix():
            dirs.append(resolved)
    # Deduplicate while preserving order
    unique: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        if d not in seen:
            unique.append(d)
            seen.add(d)
    return unique


def validate_ev_output_dir(
    output_dir: str | Path,
    *,
    cwd: Path | None = None,
    input_data_root: str | None = None,
    config_evs: str | Path | None = None,
    config_ev_tiff: str | Path | None = None,
) -> Path:
    """Return a resolved local output directory, or raise ``ForbiddenEvOutputError``.

    Cloud URI outputs are not supported for rebuild writes; they are rejected
    with a clear error (live bucket inputs must not be overwritten).
    """
    raw = str(output_dir).strip()
    if not raw:
        raise ForbiddenEvOutputError("Output directory must be a non-empty path.")

    env_root = (
        input_data_root
        if input_data_root is not None
        else os.environ.get(INPUT_DATA_ROOT_ENV, "").strip()
    )

    if _looks_like_uri(raw):
        if _uri_has_inputs_segment(raw):
            raise ForbiddenEvOutputError(
                f"Refusing to write EV layers under bucket inputs URI: {raw!r}. "
                "Choose a local folder outside inputs/ and data/evs."
            )
        raise ForbiddenEvOutputError(
            f"EV rebuild output must be a local directory, not a URI: {raw!r}."
        )

    base = (cwd or Path.cwd()).resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()

    if _path_has_inputs_segment(resolved):
        raise ForbiddenEvOutputError(
            f"Refusing to write EV layers under an 'inputs' path: {resolved}. "
            "Use a folder that is not inside inputs/."
        )

    for protected in current_evs_dirs(
        cwd=base,
        config_evs=config_evs,
        config_ev_tiff=config_ev_tiff,
    ):
        if resolved == protected or _is_under_or_equal(resolved, protected):
            raise ForbiddenEvOutputError(
                f"Refusing to write EV layers into the current EV directory "
                f"({protected}). Choose a separate output folder so the live "
                f"layer set is not overwritten."
            )

    if env_root:
        if _looks_like_uri(env_root):
            # Local output cannot equal a gs:// root; still refuse if caller
            # somehow passed a path that mirrors the same tree via relative form.
            if _uri_has_inputs_segment(env_root):
                # Already covered for URI outputs; for local paths under a
                # checkout that happens to contain "inputs" we already check
                # path segments. Nothing further for URI roots vs local dirs.
                pass
        else:
            root_path = Path(env_root).expanduser()
            if not root_path.is_absolute():
                root_path = base / root_path
            root_resolved = root_path.resolve()
            if resolved == root_resolved or _is_under_or_equal(resolved, root_resolved):
                raise ForbiddenEvOutputError(
                    f"Refusing to write EV layers under {INPUT_DATA_ROOT_ENV} "
                    f"({root_resolved}). That root is the live published input set."
                )

    return resolved


def forbidden_output_summary() -> str:
    """Human-readable list of refused destinations (for docs / --help)."""
    return (
        "Refuses: paths under inputs/; the current EV dir (data/evs and the "
        "configured paths.ev_tiff parent when it is that set); "
        f"anything under ${INPUT_DATA_ROOT_ENV}; and gs://…/inputs URIs."
    )
