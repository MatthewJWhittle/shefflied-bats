"""Registry of ``sdm data`` rebuild steps for ``--skip`` / ``--only``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional


@dataclass(frozen=True)
class RebuildStep:
    """One ordered rebuild step with a short CLI name."""

    name: str
    recorder_key: str
    summary: str


# Order matches ``rebuild_environmental_layers`` execution order.
REBUILD_STEPS: tuple[RebuildStep, ...] = (
    RebuildStep("terrain", "generate_terrain_data", "EA LIDAR DTM/DSM → terrain/"),
    RebuildStep(
        "terrain_stats",
        "generate_terrain_stats",
        "Derived terrain stats → terrain_stats.tif",
    ),
    RebuildStep("climate", "generate_climate_data", "WorldClim → climate/"),
    RebuildStep("ceh", "generate_ceh_lc_data", "CEH land cover → landcover/"),
    RebuildStep("vom", "generate_vom_data", "Vegetation Object Model → vom/"),
    RebuildStep(
        "coastal",
        "generate_coastal_distance",
        "Coastal distance → coastal_distance.tif",
    ),
    RebuildStep(
        "os",
        "process_os_data",
        "OS Vector Map → os-feature-cover.tif / os-distance-to-feature.tif",
    ),
    RebuildStep("merge", "merge_ev_layers", "Merge layers → evs-to-model.tif"),
)

REBUILD_STEP_NAMES: frozenset[str] = frozenset(step.name for step in REBUILD_STEPS)


class UnknownRebuildStepError(ValueError):
    """CLI / caller named a step that is not in the registry."""


def valid_step_names_message() -> str:
    """Human-readable list of registry names (stable order)."""
    return ", ".join(step.name for step in REBUILD_STEPS)


def parse_step_name_list(raw: Optional[str]) -> FrozenSet[str]:
    """Parse a comma-separated step list; empty/None → empty set.

    Raises
    ------
    UnknownRebuildStepError
        If any name is not in :data:`REBUILD_STEP_NAMES`.
    """
    if raw is None:
        return frozenset()
    names = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not names:
        return frozenset()
    unknown = names - REBUILD_STEP_NAMES
    if unknown:
        raise UnknownRebuildStepError(
            "Unknown rebuild step(s): "
            + ", ".join(sorted(unknown))
            + f". Valid names: {valid_step_names_message()}"
        )
    return frozenset(names)


def resolve_active_steps(
    *,
    skip: Iterable[str] = (),
    only: Iterable[str] = (),
) -> FrozenSet[str]:
    """Return the set of step names that should run.

    ``skip`` and ``only`` are mutually exclusive (non-empty together raises
    ``ValueError``). Names must already be validated registry members.
    """
    skip_set = frozenset(skip)
    only_set = frozenset(only)
    if skip_set and only_set:
        raise ValueError("Use only one of --skip and --only")
    unknown = (skip_set | only_set) - REBUILD_STEP_NAMES
    if unknown:
        raise UnknownRebuildStepError(
            "Unknown rebuild step(s): "
            + ", ".join(sorted(unknown))
            + f". Valid names: {valid_step_names_message()}"
        )
    if only_set:
        return only_set
    return REBUILD_STEP_NAMES - skip_set


__all__ = [
    "REBUILD_STEPS",
    "REBUILD_STEP_NAMES",
    "RebuildStep",
    "UnknownRebuildStepError",
    "parse_step_name_list",
    "resolve_active_steps",
    "valid_step_names_message",
]
