import logging
from pathlib import Path

from sdm.utils.logging_utils import setup_logging
# from species_sdm.data.extraction import extract_ev_data_at_points # To be uncommented
# from species_sdm.models.preprocessor import scale_features # To be uncommented

def extract_features(
    input_points_path: Path,
    ev_raster_stack_path: Path,
    output_path: Path = Path("data/processed/model_input_features.parquet"),
    verbose: bool = False
) -> None:
    """
    Core function to extract environmental variable data for points.
    Can be called from other scripts or notebooks.

    Args:
        input_points_path: Path to combined occurrence and background points data
        ev_raster_stack_path: Path to the multi-band environmental variable raster stack
        output_path: Path to save the final model-ready feature data
        verbose: Enable verbose logging
    """
    setup_logging(verbose=verbose)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Extracting features from: {ev_raster_stack_path} for points in: {input_points_path}")

    # 1. Load points (GeoDataFrame expected)
    # points_gdf = gpd.read_parquet(input_points_path) # Or other format as needed

    # 2. Call extraction function (placeholder)
    # temp_extracted_path = output_path.with_suffix(".temp_extracted.parquet")
    # extract_ev_data_at_points(points_gdf, ev_raster_stack_path, temp_extracted_path)
    
    # 3. Load extracted data
    # features_df = pd.read_parquet(temp_extracted_path)

    # 4. Call preprocessing/scaling function (placeholder)
    # if columns_to_scale:
    #    features_df = scale_features(features_df, columns_to_scale)
    
    # 5. Save final data
    # features_df.to_parquet(output_path)
    logging.warning("Feature extraction and preprocessing logic not yet implemented. Please extract from notebook.")
    logging.info(f"Model-ready feature data (placeholder) would be saved to: {output_path}")

    # Optionally, clean up temporary files like temp_extracted_path
    # if temp_extracted_path.exists():
    #    temp_extracted_path.unlink() 