"""Export GeoTIFFs for sharing: optional CRS change and optional COG encoding (orthogonal flags)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from rasterio.enums import Resampling

from sdm.raster.io import export_geotiff

logger = logging.getLogger(__name__)

_RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


def _output_suffix(output_crs: Optional[str], as_cog: bool) -> str:
    parts: List[str] = []
    if output_crs:
        parts.append(output_crs.replace(":", "").replace(" ", ""))
    if as_cog:
        parts.append("cog")
    return ("_" + "_".join(parts)) if parts else ""


def export_raster_paths(
    raster_paths: Iterable[Path],
    output_dir: Path,
    *,
    output_crs: Optional[str] = None,
    as_cog: bool = False,
    resampling: str = "bilinear",
    quiet_cog: bool = True,
) -> List[Path]:
    """Export each raster into *output_dir* with suffix from CRS / COG choices.

    Raises:
        ValueError: If neither ``output_crs`` nor ``as_cog`` is set (nothing to do).
        FileNotFoundError: If an input path is missing.
    """
    if output_crs is None and not as_cog:
        raise ValueError(
            "Nothing to do: pass output_crs (reproject) and/or as_cog=True (COG encode)."
        )

    try:
        rs = _RESAMPLING_MAP[resampling.lower()]
    except KeyError as e:
        raise ValueError(
            f"Unknown resampling {resampling!r}; choose from {sorted(_RESAMPLING_MAP)}"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _output_suffix(output_crs, as_cog)
    written: List[Path] = []
    stem_counts: dict[str, int] = {}

    for src in raster_paths:
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"Raster not found: {src}")

        base_key = f"{src.stem}{suffix}"
        idx = stem_counts.get(base_key, 0)
        stem_counts[base_key] = idx + 1
        disambig = "" if idx == 0 else f"_{idx + 1}"
        dst = output_dir / f"{src.stem}{suffix}{disambig}.tif"
        logger.info(
            "Export %s -> %s (output_crs=%s, as_cog=%s)",
            src,
            dst,
            output_crs or "(source)",
            as_cog,
        )
        export_geotiff(
            src,
            dst,
            dst_crs=output_crs,
            as_cog=as_cog,
            resampling=rs,
            quiet=quiet_cog,
        )
        written.append(dst)

    return written
