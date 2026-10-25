"""Resolved background-point settings for the live ``sdm train`` path.

Knobs cover surface generation, per-model preparation (exclusion, thinning,
cap), distance sample weights, class balance, and seed. Defaults match today's
live-train behaviour so an unchanged config stays byte-identical.

Merge order (later wins): code defaults → ``config.yml`` ``background:`` →
``model_config.yml`` ``model.background`` / reconciled ``model.sampling`` →
CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Optional, Union

from sdm.occurrence.sampling import BackgroundMethod, TransformMethod
from sdm.types import (
    BackgroundConfig,
    ModelConfig,
    ProjectConfig,
    SamplingConfig,
)


class BackgroundSurface(StrEnum):
    """Where shared background points are drawn from."""

    ALL_BAT = "all_bat"
    UNIFORM = "uniform"


# Live-train defaults (must match current main when no overrides are set).
DEFAULT_N_BACKGROUND_POINTS = 8000
DEFAULT_BACKGROUND_VALUE = 0.025
DEFAULT_SIGMA = 6.5
DEFAULT_PRESENCE_EXCLUSION_M = 500.0
DEFAULT_THIN_M = 2000.0
DEFAULT_CAP_RATIO = 10.0
DEFAULT_SAMPLE_WEIGHT_N_NEIGHBORS = 10  # live CLI default; YAML previously said 5


@dataclass(frozen=True)
class ResolvedBackgroundSettings:
    """Fully resolved background settings for one training run."""

    n_background_points: int
    surface: BackgroundSurface
    background_method: BackgroundMethod
    background_value: float
    sigma: float
    transform_method: TransformMethod
    surface_resolution_m: int
    presence_exclusion_m: float
    presence_exclusion_max_m: float
    background_thin_m: float
    presence_thin_m: float
    cap_ratio: Optional[float]
    min_background: Optional[int]
    max_background: Optional[int]
    distance_weights: bool
    background_distance_weights: bool
    sample_weight_n_neighbors: int
    class_balance: bool
    seed: int

    def to_metadata(self) -> dict[str, Any]:
        """Every resolved knob for ``run_metadata.json`` / published run.json."""
        return {
            "n_background_points": int(self.n_background_points),
            "surface": self.surface.value,
            "background_method": self.background_method.value,
            "background_value": float(self.background_value),
            "sigma": float(self.sigma),
            "transform_method": self.transform_method.value,
            "surface_resolution_m": int(self.surface_resolution_m),
            "presence_exclusion_m": float(self.presence_exclusion_m),
            "presence_exclusion_max_m": (
                None
                if self.presence_exclusion_max_m == float("inf")
                else float(self.presence_exclusion_max_m)
            ),
            "background_thin_m": float(self.background_thin_m),
            "presence_thin_m": float(self.presence_thin_m),
            "cap_ratio": (
                None if self.cap_ratio is None else float(self.cap_ratio)
            ),
            "min_background": self.min_background,
            "max_background": self.max_background,
            "distance_weights": bool(self.distance_weights),
            "background_distance_weights": bool(self.background_distance_weights),
            "sample_weight_n_neighbors": int(self.sample_weight_n_neighbors),
            "class_balance": bool(self.class_balance),
            "seed": int(self.seed),
        }

    def maxent_class_weights(self) -> Union[str, float]:
        """Value passed to MaxEnt: ``\"balanced\"`` or ``1.0`` when balance is off."""
        return "balanced" if self.class_balance else 1.0

    def n_max_background(self, n_presence: int) -> Optional[int]:
        """Absolute background cap for one model, or ``None`` when uncapped."""
        if self.cap_ratio is None and self.min_background is None and self.max_background is None:
            return None
        n: Optional[int] = None
        if self.cap_ratio is not None:
            n = int(n_presence * self.cap_ratio)
        if self.min_background is not None:
            n = self.min_background if n is None else max(n, self.min_background)
        if self.max_background is not None:
            n = self.max_background if n is None else min(n, self.max_background)
        return n


def _as_background_method(value: Any) -> BackgroundMethod:
    if isinstance(value, BackgroundMethod):
        return value
    return BackgroundMethod(str(value).lower())


def _as_transform_method(value: Any) -> TransformMethod:
    if isinstance(value, TransformMethod):
        return value
    return TransformMethod(str(value).lower())


def _as_surface(value: Any) -> BackgroundSurface:
    if isinstance(value, BackgroundSurface):
        return value
    return BackgroundSurface(str(value).lower())


def _overlay_mapping(
    base: dict[str, Any],
    overlay: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply overlay keys onto base.

    ``None`` is applied for nullable knobs (``cap_ratio``, absolute caps,
    optional max exclusion / surface resolution) so YAML ``null`` turns them
    off. Other keys skip ``None`` so partial overlays do not wipe defaults.
    """
    if not overlay:
        return base
    nullable = {
        "cap_ratio",
        "min_background",
        "max_background",
        "presence_exclusion_max_m",
        "surface_resolution_m",
    }
    out = dict(base)
    for key, value in overlay.items():
        if value is not None or key in nullable:
            out[key] = value
    return out


def _background_config_to_dict(cfg: Optional[BackgroundConfig]) -> dict[str, Any]:
    """Dump only fields explicitly set in YAML (plus nullable clears).

    Avoids pydantic defaults on a partial ``background:`` block from wiping
    values already resolved from ``sampling.grid_size_m`` / code defaults.
    """
    if cfg is None:
        return {}
    raw = cfg.model_dump(exclude_none=False)
    raw.pop("profile", None)
    fields_set = set(getattr(cfg, "model_fields_set", set()))
    # Always allow nullable clears when the field was named in YAML.
    return {k: v for k, v in raw.items() if k in fields_set}


def _sampling_legacy_overlay(sampling: Optional[SamplingConfig]) -> dict[str, Any]:
    """Map legacy ``model.sampling`` keys onto the unified background knobs.

    Live train previously ignored YAML ``sample_weight_n_neighbors: 5`` (CLI
    default 10 won) and ``sampling.background`` min/max. Those ignored min/max
    values must not suddenly apply — absolute caps live under
    ``background.min_background`` / ``max_background`` (default null).
    ``sampling.background.factor`` maps to ``cap_ratio`` when the latter is
    unset. Thinning distances come from ``sampling.grid_size_m``.
    """
    if sampling is None:
        return {}
    out: dict[str, Any] = {
        "presence_thin_m": float(sampling.grid_size_m),
        "background_thin_m": float(sampling.grid_size_m),
    }
    bg = sampling.background
    if bg is not None:
        out["cap_ratio"] = float(bg.factor)
    return out


def default_background_dict(*, surface_resolution_m: int = 100) -> dict[str, Any]:
    """Code defaults identical to current main live-train behaviour."""
    return {
        "n_background_points": DEFAULT_N_BACKGROUND_POINTS,
        "surface": BackgroundSurface.ALL_BAT,
        "background_method": BackgroundMethod.CONTRAST,
        "background_value": DEFAULT_BACKGROUND_VALUE,
        "sigma": DEFAULT_SIGMA,
        "transform_method": TransformMethod.PRESENCE,
        "surface_resolution_m": int(surface_resolution_m),
        "presence_exclusion_m": DEFAULT_PRESENCE_EXCLUSION_M,
        "presence_exclusion_max_m": float("inf"),
        "background_thin_m": DEFAULT_THIN_M,
        "presence_thin_m": DEFAULT_THIN_M,
        "cap_ratio": DEFAULT_CAP_RATIO,
        "min_background": None,
        "max_background": None,
        "distance_weights": True,
        "background_distance_weights": True,
        "sample_weight_n_neighbors": DEFAULT_SAMPLE_WEIGHT_N_NEIGHBORS,
        "class_balance": True,
    }


def resolve_background_settings(
    *,
    project_config: Optional[ProjectConfig] = None,
    model_config: Optional[ModelConfig] = None,
    seed: Optional[int] = None,
    n_background_points_override: Optional[int] = None,
    presence_exclusion_m_override: Optional[float] = None,
    presence_exclusion_max_m_override: Optional[float] = None,
    sample_weight_n_neighbors_override: Optional[int] = None,
    surface_resolution_m: Optional[int] = None,
) -> ResolvedBackgroundSettings:
    """Merge project, model, and CLI overrides into resolved settings."""
    resolution = 100
    if surface_resolution_m is not None:
        resolution = int(surface_resolution_m)
    elif project_config is not None:
        resolution = int(project_config.spatial.resolution)

    merged = default_background_dict(surface_resolution_m=resolution)

    project_bg = getattr(project_config, "background", None) if project_config else None
    merged = _overlay_mapping(merged, _background_config_to_dict(project_bg))

    model_cfg = model_config
    if model_cfg is not None:
        # Legacy sampling keys first; explicit model.background wins over them.
        merged = _overlay_mapping(merged, _sampling_legacy_overlay(model_cfg.sampling))
        merged = _overlay_mapping(merged, _background_config_to_dict(model_cfg.background))

    if n_background_points_override is not None:
        merged["n_background_points"] = int(n_background_points_override)
    if presence_exclusion_m_override is not None:
        merged["presence_exclusion_m"] = float(presence_exclusion_m_override)
    if presence_exclusion_max_m_override is not None:
        merged["presence_exclusion_max_m"] = float(presence_exclusion_max_m_override)
    if sample_weight_n_neighbors_override is not None:
        merged["sample_weight_n_neighbors"] = int(sample_weight_n_neighbors_override)

    resolved_seed = seed
    if resolved_seed is None and model_cfg is not None:
        resolved_seed = model_cfg.seed
    if resolved_seed is None:
        resolved_seed = 42

    cap_ratio = merged.get("cap_ratio")
    if cap_ratio is not None:
        cap_ratio = float(cap_ratio)

    min_bg = merged.get("min_background")
    max_bg = merged.get("max_background")
    if min_bg is not None:
        min_bg = int(min_bg)
    if max_bg is not None:
        max_bg = int(max_bg)

    d_max = merged.get("presence_exclusion_max_m", float("inf"))
    if d_max is None:
        d_max = float("inf")
    else:
        d_max = float(d_max)

    surf_res = merged.get("surface_resolution_m")
    if surf_res is None:
        surf_res = resolution
    else:
        surf_res = int(surf_res)

    return ResolvedBackgroundSettings(
        n_background_points=int(merged["n_background_points"]),
        surface=_as_surface(merged["surface"]),
        background_method=_as_background_method(merged["background_method"]),
        background_value=float(merged["background_value"]),
        sigma=float(merged["sigma"]),
        transform_method=_as_transform_method(merged["transform_method"]),
        surface_resolution_m=surf_res,
        presence_exclusion_m=float(merged["presence_exclusion_m"]),
        presence_exclusion_max_m=d_max,
        background_thin_m=float(merged["background_thin_m"]),
        presence_thin_m=float(merged["presence_thin_m"]),
        cap_ratio=cap_ratio,
        min_background=min_bg,
        max_background=max_bg,
        distance_weights=bool(merged["distance_weights"]),
        background_distance_weights=bool(merged["background_distance_weights"]),
        sample_weight_n_neighbors=int(merged["sample_weight_n_neighbors"]),
        class_balance=bool(merged["class_balance"]),
        seed=int(resolved_seed),
    )
