import logging
from pathlib import Path
from typing import Union # Not strictly needed

import typer
import xarray as xr
import rioxarray as rxr # For direct rio operations if needed, though utils are preferred
import numpy as np
from rasterio.enums import Resampling # For specifying resampling method

from sdm.utils.logging_utils import setup_logging
from sdm.utils.io import load_boundary_and_transform
from sdm.data.landcover import get_ceh_land_cover_codes_v2023, define_broad_habitat_categories
from sdm.raster.processing import create_binary_raster_from_category, aggregate_categorical_rasters
# The reproject_data utility can be used if its parameterization fits.
# Original script used lc_processed.rio.reproject directly with resampling=0 (NearestNeighbor).
# Our reproject_data uses Resampling.bilinear by default, but accepts a resampling arg.
from sdm.raster.utils import reproject_data, squeeze_dataset 

app = typer.Typer()

@app.command()
def main(
    output_dir: Path = typer.Option(
        "data/evs/landcover", 
        help="Directory to save the output CEH land cover GeoTIFF.",
        writable=True, resolve_path=True
    ),
    boundary_path: Path = typer.Option(
        "data/processed/boundary.geojson", 
        help="Path to the boundary file for clipping and context.",
        exists=True, readable=True, resolve_path=True
    ),
    ceh_data_path: Path = typer.Option(
        "data/raw/big-files/CEH/data/7727ce7d-531e-4d77-b756-5cc59ff016bd/gblcm2023_10m.tif",
        help="Path to the raw CEH land cover GeoTIFF file (e.g., gblcm2023_10m.tif).",
        exists=True, readable=True, resolve_path=True
    ),
    buffer_distance_m: float = typer.Option(1000, help="Buffer distance in meters for the boundary when clipping raw data."),
    # Original script had `resolution` (input data res) and `coarsen_factor`.
    # It's clearer to specify final target resolution for the EV.
    # Assume input CEH is 10m. If target is 100m, coarsen_factor is 10.
    output_resolution_m: int = typer.Option(100, help="Target output resolution in meters for the processed land cover EV."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
) -> None:
    """
    Processes CEH land cover data to generate derived environmental variables.

    Steps:
    1. Load boundary and spatial configuration.
    2. Load raw CEH land cover data.
    3. Clip CEH data to the (buffered) study boundary.
    4. Convert raw land cover codes into binary layers for each category.
    5. Coarsen these binary layers to the target output resolution, summing values (effectively calculating area or count).
    6. Aggregate specific land cover categories into broader habitat types.
    7. Reproject the final processed dataset to the model's reference CRS.
    8. Save the multi-band GeoTIFF.
    """
    setup_logging(verbose=verbose)
    logging.info(f"Starting CEH Land Cover processing. Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load boundary and project's spatial parameters (CRS, transform for final output grid)
    # We need the boundary_gdf for clipping, and model_crs/transform/resolution for final reprojection.
    boundary_gdf, model_transform, _, spatial_config = load_boundary_and_transform(
        boundary_path, buffer_distance=0 # Buffer for clipping done separately later
    )
    model_crs = boundary_gdf.crs # This is the target CRS for the final EV
    # model_resolution = spatial_config["resolution"] # This is the project's default EV res.
                                                # For this script, output_resolution_m is the key target.

    logging.info(f"Loading CEH land cover data from: {ceh_data_path}")
    try:
        land_cover_raw = rxr.open_rasterio(ceh_data_path).squeeze(drop=True)
    except Exception as e:
        logging.error(f"Failed to load CEH data from {ceh_data_path}: {e}")
        raise typer.Exit(code=1)
    
    # Determine input resolution for coarsen factor calculation
    input_res_x, _ = land_cover_raw.rio.resolution()
    input_resolution_m = abs(int(input_res_x))
    if input_resolution_m == 0:
        logging.error("Could not determine input resolution of CEH data, or it is zero.")
        raise typer.Exit(code=1)
    
    if output_resolution_m < input_resolution_m:
        logging.error(f"Target output resolution {output_resolution_m}m must be coarser than or equal to input {input_resolution_m}m.")
        raise typer.Exit(code=1)
    
    coarsen_factor = output_resolution_m // input_resolution_m
    if coarsen_factor == 0 : coarsen_factor = 1 # if same resolution, factor is 1

    logging.info(f"Clipping CEH data to boundary (buffer: {buffer_distance_m}m)...")
    # Reproject boundary to CRS of CEH data for clipping
    boundary_for_clipping = boundary_gdf.to_crs(land_cover_raw.rio.crs)
    boundary_for_clipping["geometry"] = boundary_for_clipping.geometry.buffer(buffer_distance_m)
    
    try:
        lc_clipped = land_cover_raw.rio.clip_box(*boundary_for_clipping.total_bounds, crs=boundary_for_clipping.crs)
    except Exception as e:
        logging.error(f"Failed to clip CEH data: {e}. Check boundary and CEH data CRS and overlap.")
        raise typer.Exit(code=1)

    # Handle nodata after clipping
    if lc_clipped.rio.nodata is not None:
        lc_clipped = lc_clipped.where(lc_clipped != lc_clipped.rio.nodata)
    # else: assume no explicit nodata, or handle based on expected fill value if known
    # For CEH data, 0 is often a valid background/unclassified, not necessarily nodata.
    # Let's assume NaN is the internal representation of nodata for processing.
    lc_clipped.rio.write_nodata(np.nan, inplace=True)

    logging.info("Creating binary layers for each CEH land cover category...")
    ceh_codes = get_ceh_land_cover_codes_v2023()
    category_layers = []
    for code_str, label in ceh_codes.items():
        try:
            category_val = int(code_str)
            binary_layer_ds = create_binary_raster_from_category(
                source_raster=lc_clipped, 
                category_value=category_val,
                output_var_name=label.replace(" ", "_").replace(",", "").replace("/", "_") # Sanitize label for var name
            )
            category_layers.append(binary_layer_ds)
        except ValueError:
            logging.warning(f"Invalid code '{code_str}' in CEH mapping. Skipping.")
    
    if not category_layers:
        logging.error("No category layers were created. Check CEH codes and data.")
        raise typer.Exit(code=1)
    
    lc_stacked_categories = xr.merge(category_layers)

    logging.info(f"Coarsening category layers from {input_resolution_m}m to {output_resolution_m}m (factor {coarsen_factor}) and summing...")
    # The sum after coarsen effectively gives count of original pixels, or area if multiplied by pixel_area before.
    # Original script did lc_stack * area_per_pixel, then coarsen.sum().
    # This means the sum is total area of that category in the coarser pixel.
    pixel_area_input_res = input_resolution_m * input_resolution_m
    lc_area_stacked = lc_stacked_categories * pixel_area_input_res
    
    if coarsen_factor > 1:
        lc_coarsened_sum = lc_area_stacked.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").sum(skipna=True) # skipna=True is safer
    else: # No coarsening needed if factor is 1
        lc_coarsened_sum = lc_area_stacked
    # Resulting values are area (e.g. m^2) of each category in pixels of size output_resolution_m

    logging.info("Aggregating into broad habitat categories...")
    broad_habitat_map = define_broad_habitat_categories()
    # Ensure variable names in lc_coarsened_sum match keys in broad_habitat_map (sanitized labels)
    lc_processed = aggregate_categorical_rasters(
        categorical_raster_ds=lc_coarsened_sum, 
        aggregation_map=broad_habitat_map,
        # Original script dropped ["Inland rock", "Marine, Littoral", "Freshwater"]
        # My define_broad_habitat_categories created "Other_Bare", "Coastal_Marine", and grouped "Freshwater" into "Wetland_Bog"
        # Let's make the drop explicit if desired, using the new aggregated names.
        categories_to_drop=["Other_Bare", "Coastal_Marine"] # Example: if these broad cats are not needed
    )

    logging.info(f"Reprojecting processed CEH data to target CRS: {model_crs} at {output_resolution_m}m...")
    # We need a transform for the target grid at output_resolution_m using model_crs
    # load_boundary_and_transform provided `model_transform` for `spatial_config["resolution"]`
    # We need one for `output_resolution_m`
    # For simplicity, if output_resolution_m matches spatial_config["resolution"], use model_transform.
    # Otherwise, it's safer to let rio.reproject calculate it, or create a new one.
    # The original script created a new transform for its target_resolution.
    # Let's use the `reproject_data` utility which handles transform creation if not given one that matches resolution.
    # However, reproject_data needs a full Affine transform if provided.
    # The original: lc_processed.rio.reproject(spatial_config["crs"], transform=model_transform, resampling=0)
    # This implies model_transform was for the target_resolution of that step.

    # Re-create transform for the specific output_resolution_m and the boundary_gdf's extent
    # This step requires careful alignment with the project's master grid if one exists.
    # Using load_boundary_and_transform again with the *output_resolution_m* is one way,
    # but it re-reads boundary. Let's try to construct one or use rio.reproject without explicit transform.

    final_lc_ds = lc_processed.rio.reproject(
        dst_crs=model_crs, 
        resolution=output_resolution_m, 
        resampling=Resampling.nearest # Categorical data needs nearest neighbor
    )
    final_lc_ds = squeeze_dataset(final_lc_ds) # If reproject creates extra dims

    output_filename = f"ceh_landcover_processed_{output_resolution_m}m.tif"
    output_path = output_dir / output_filename
    logging.info(f"Writing final processed CEH land cover data to: {output_path}")
    final_lc_ds.rio.to_raster(output_path)

    logging.info("CEH Land Cover data processing finished.")

if __name__ == "__main__":
    app() 