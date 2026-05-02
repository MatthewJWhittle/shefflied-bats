"""
Split a multi-band raster into separate single-band rasters.
"""

import logging
from pathlib import Path
from typing import Optional

import rasterio as rio
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_raster_by_band(
    input_raster: Path,
    output_dir: Path,
    output_prefix: Optional[str] = None,
    use_band_names: bool = True,
    window_size: int = 1024,
) -> list[Path]:
    """Split a multi-band raster into separate single-band rasters.
    
    Args:
        input_raster: Path to the input multi-band raster
        output_dir: Directory to write output single-band rasters
        output_prefix: Optional prefix for output filenames. If None, uses input filename stem.
        use_band_names: If True, use band descriptions/names in output filenames. If False, use band numbers.
        window_size: Size of processing windows for efficient I/O
        
    Returns:
        List of paths to the created output rasters
    """
    input_raster = Path(input_raster)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_prefix is None:
        output_prefix = input_raster.stem
    
    logger.info(f"Splitting raster {input_raster} into separate bands...")
    
    output_paths = []
    
    with rio.open(input_raster) as src:
        num_bands = src.count
        logger.info(f"Found {num_bands} bands in input raster")
        
        # Create windows for efficient processing
        if window_size > 0:
            # Create custom windows based on window_size
            height = src.height
            width = src.width
            n_windows_h = (height + window_size - 1) // window_size
            n_windows_w = (width + window_size - 1) // window_size
            
            windows = []
            for i in range(n_windows_h):
                for j in range(n_windows_w):
                    row_off = i * window_size
                    col_off = j * window_size
                    win_height = min(window_size, height - row_off)
                    win_width = min(window_size, width - col_off)
                    windows.append(rio.windows.Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=win_width,
                        height=win_height
                    ))
        else:
            # Use block windows from the raster
            windows = list(src.block_windows(1))
        
        # Process each band
        for band_idx in range(1, num_bands + 1):
            # Get band name/description
            band_name = src.descriptions[band_idx - 1] if src.descriptions else None
            
            # Generate output filename
            if use_band_names and band_name:
                # Sanitize band name for filesystem
                safe_name = band_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                output_filename = f"{output_prefix}_{safe_name}.tif"
            else:
                output_filename = f"{output_prefix}_band_{band_idx}.tif"
            
            output_path = output_dir / output_filename
            output_paths.append(output_path)
            
            # Create output profile for single-band raster
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'compress': 'deflate',
            })
            
            logger.info(f"Writing band {band_idx} to {output_path}")
            
            with rio.open(output_path, 'w', **profile) as dst:
                # Set band description
                if band_name:
                    dst.descriptions = [band_name]
                
                # Copy data window by window
                for window in tqdm(windows, desc=f"Band {band_idx}", leave=False):
                    data = src.read(band_idx, window=window)
                    dst.write(data, 1, window=window)
    
    logger.info(f"Successfully split raster into {len(output_paths)} files")
    return output_paths





