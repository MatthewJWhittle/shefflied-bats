import logging
from pathlib import Path
import typer
from typing_extensions import Annotated

from sdm.utils.logging_utils import setup_logging
# from species_sdm.data.extraction import extract_ev_data_at_points # To be uncommented
# from species_sdm.models.preprocessor import scale_features # To be uncommented

app = typer.Typer()

@app.command()
def main(
    input_points_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to combined occurrence and background points data (e.g., Parquet from generate_background_points.py).",
            exists=True, readable=True, resolve_path=True
        )
    ],
    ev_raster_stack_path: Annotated[
        Path, 
        typer.Option(
            ..., # Required
            help="Path to the multi-band environmental variable raster stack (e.g., GeoTIFF from merge_ev_layers.py).",
            exists=True, readable=True, resolve_path=True
        )
    ],
    output_path: Annotated[
        Path, 
        typer.Option(
            help="Path to save the final model-ready feature data (e.g., Parquet).",
            writable=True, resolve_path=True
        )
    ] = Path("data/processed/model_input_features.parquet"),
    # columns_to_scale: Annotated[Optional[List[str]], typer.Option(help="List of columns to scale (if preprocessing needed).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")] = False
) -> None:
    """
    Extracts environmental variable (EV) data for occurrence/background points,
    applies any necessary preprocessing (e.g., scaling), and saves the 
    model-ready feature dataset.
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

if __name__ == "__main__":
    app() 