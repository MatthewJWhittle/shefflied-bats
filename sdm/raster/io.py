from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
import os
import shutil
import tempfile

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

import xarray as xr
import rioxarray as rxr

logger = logging.getLogger(__name__)


def _deflate_profile_for_dtype(dtype: Union[np.dtype, str]) -> Dict[str, Any]:
    """Deflate COG kwargs matched to raster dtype (float → predictor 3)."""
    kw = cog_profiles.get("deflate").copy()
    dt = np.dtype(dtype)
    kw.update(dtype=np.dtype(dtype).name)
    if np.issubdtype(dt, np.floating):
        kw["predictor"] = 3
    elif np.issubdtype(dt, np.integer):
        kw["predictor"] = 2
    else:
        kw["predictor"] = 1
    return kw


def _resolve_target_crs(
    dst_crs: Optional[Union[str, int, CRS]],
    src_crs: Optional[CRS],
) -> CRS:
    if dst_crs is None:
        if src_crs is None:
            raise ValueError("Source raster has no CRS; pass dst_crs explicitly.")
        return src_crs
    if isinstance(dst_crs, CRS):
        return dst_crs
    if isinstance(dst_crs, int):
        return CRS.from_epsg(dst_crs)
    return CRS.from_user_input(dst_crs)


def export_geotiff(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    *,
    dst_crs: Optional[Union[str, int, CRS]] = None,
    as_cog: bool = False,
    resampling: Resampling = Resampling.bilinear,
    quiet: bool = True,
) -> None:
    """Optionally reproject and/or encode a GeoTIFF as a COG.

    ``dst_crs`` and ``as_cog`` are independent: set either or both.

    * ``dst_crs=None`` keeps the source CRS/grid (no warp).
    * ``as_cog=False`` writes a plain tiled/deflate GeoTIFF when warping; otherwise copies the file when nothing else applies.

    Args:
        src_path: Input raster path.
        dst_path: Output path (must differ from ``src_path``).
        dst_crs: Target CRS (e.g. ``EPSG:3857`` or ``3857``). ``None`` = match source grid.
        as_cog: When True, final output is Cloud Optimized GeoTIFF (via ``rio-cogeo``).
        resampling: Warp resampling when ``dst_crs`` differs from the source CRS.
        quiet: Passed through to ``cog_translate``.
    """
    src_path = Path(src_path).resolve()
    dst_path = Path(dst_path).resolve()
    if src_path == dst_path:
        raise ValueError(f"Destination must differ from source ({src_path})")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {src_path}")
        target_crs = _resolve_target_crs(dst_crs, src.crs)
        needs_reproject = target_crs != src.crs

        if needs_reproject:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *array_bounds(src.height, src.width, src.transform),
            )
            meta = src.meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "crs": target_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "compress": "deflate",
                    "tiled": True,
                    "BIGTIFF": "IF_NEEDED",
                }
            )

            fd, tmp_name = tempfile.mkstemp(suffix=".tif", dir=str(dst_path.parent))
            os.close(fd)
            warped_path = Path(tmp_name)
            try:
                with rio.open(warped_path, "w", **meta) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rio.band(src, i),
                            destination=rio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=target_crs,
                            resampling=resampling,
                            src_nodata=src.nodata,
                            dst_nodata=src.nodata,
                        )
                    dst.descriptions = src.descriptions

                if as_cog:
                    cog_kw = _deflate_profile_for_dtype(meta["dtype"])
                    cog_translate(
                        warped_path,
                        dst_path,
                        cog_kw,
                        nodata=meta.get("nodata"),
                        overview_resampling="nearest",
                        forward_band_tags=True,
                        quiet=quiet,
                    )
                else:
                    shutil.move(str(warped_path), str(dst_path))
                    warped_path = None  # suppress unlink in finally
            finally:
                if warped_path is not None:
                    warped_path.unlink(missing_ok=True)
            return

        # Same CRS/grid as source
        if as_cog:
            cog_kw = _deflate_profile_for_dtype(src.profile["dtype"])
            cog_translate(
                src_path,
                dst_path,
                cog_kw,
                nodata=src.nodata,
                overview_resampling="nearest",
                forward_band_tags=True,
                quiet=quiet,
            )
            return

        shutil.copy2(src_path, dst_path)


def cogify_geotiff_inplace(path: Union[str, Path], *, quiet: bool = True) -> None:
    """Rewrite *path* as a Cloud Optimized GeoTIFF (same CRS, grid, dtype)."""
    path = Path(path).resolve()
    with rio.open(path) as src:
        nodata = src.nodata
        cog_kw = _deflate_profile_for_dtype(src.profile["dtype"])
    fd, tmp_name = tempfile.mkstemp(suffix=".tif", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        cog_translate(
            path,
            tmp_path,
            cog_kw,
            nodata=nodata,
            overview_resampling="nearest",
            forward_band_tags=True,
            quiet=quiet,
        )
        path.unlink(missing_ok=True)
        tmp_path.rename(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def translate_to_cog(
    src_path: Path, 
    dst_path: Path, 
    profile: str = "webp", # common default for good compression/quality 
    profile_options: Optional[Dict[str, Any]] = None, # Changed to Optional[Dict]
    **options: Any # Pass other rio_cogeo options directly
) -> None:
    """Translates a raster to a Cloud Optimized GeoTIFF (COG).

    Args:
        src_path: Path to the source raster file.
        dst_path: Path to save the output COG file.
        profile: COG profile to use (e.g., "jpeg", "webp", "zstd", "lzw").
                 See rio-cogeo documentation for available profiles.
        profile_options: Dictionary of options for the chosen profile.
        **options: Additional keyword arguments to pass to cog_translate.
    """
    effective_profile_options = profile_options if profile_options is not None else {}

    # Get default profile options and update with any user-provided ones
    dst_profile = cog_profiles.get(profile)
    if not dst_profile:
        raise ValueError(f"Unknown COG profile: {profile}. Available: {list(cog_profiles.keys())}")
    
    # Create a mutable copy of the default profile to update
    final_dst_profile = dst_profile.copy()

    # Update default profile options with user-specified ones from effective_profile_options
    for key, value in effective_profile_options.items():
        if key in final_dst_profile:
            final_dst_profile[key] = value
        else:
            # Or, if you want to allow adding new keys to the profile dict (less safe)
            # final_dst_profile[key] = value 
            print(f"Warning: Option '{key}' not standard for profile '{profile}'. It will be passed to general cog_translate options.")
            # If an option is not part of the profile, pass it to general options
            if key not in options:
                 options[key] = value

    # Ensure output directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cog_translate(
        src_path,
        dst_path,
        final_dst_profile, # Use the merged profile dictionary
        **options,
    )
    # print(f"Successfully translated {src_path} to COG: {dst_path}") # Optional: add logging 

def load_environmental_variables(
    ev_path: Union[str, Path]
) -> Tuple[xr.Dataset, Path]:
    """Load environmental variables for modelling.
    
    Args:
        ev_path: Path to environmental variables raster
        
    Returns:
        Tuple of (Dataset containing environmental variables, Path to raster file)
    """
    ev_raster = Path(ev_path)
    
    try:
        evs : xr.Dataset = rxr.open_rasterio(ev_raster, masked=True, band_as_variable=True).squeeze() # type: ignore
        
        # rename the variables by their long name
        for var in evs.data_vars:
            evs = evs.rename({var: evs[var].attrs["long_name"]})
        logger.debug("Loaded environmental variables from %s", ev_raster)
        return evs, ev_raster
    except Exception as e:
        logger.error("Error loading environmental variables: %s", e)
        raise 