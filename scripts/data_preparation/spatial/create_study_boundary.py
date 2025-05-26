import typer
from pathlib import Path
import logging
from typing import Optional

from sdm.data.spatial import create_study_boundary
from sdm.utils.logging_utils import setup_logging

app = typer.Typer()

@app.command()
def main(
    raw_counties_file: Path = typer.Option(
        "data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson", 
        help="Path to the raw UK counties GeoJSON file.",
        exists=True, readable=True, file_okay=True
    ),
    output_geojson: Path = typer.Option(
        "data/processed/boundary.geojson", 
        help="Path to save the processed study area boundary GeoJSON."
    ),
    target_crs: str = typer.Option("EPSG:27700", help="Target CRS for the output boundary."),
    simplify_tolerance: Optional[float] = typer.Option(100.0, help="Simplification tolerance for TopoJSON (meters). Set to 0 or negative for no simplification."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
) -> None:
    """
    Creates the study area boundary GeoJSON file by loading UK counties, 
    filtering for a defined Yorkshire region, reprojecting, simplifying, 
    and dissolving.
    """
    setup_logging(verbose=verbose)
    logging.info(f"Creating study area boundary from: {raw_counties_file}")

    actual_simplify_tolerance = simplify_tolerance if simplify_tolerance and simplify_tolerance > 0 else None

    try:
        study_area_gdf = create_study_boundary(
            counties_filepath=raw_counties_file,
            target_crs=target_crs,
            simplify_tolerance=actual_simplify_tolerance
        )
    except FileNotFoundError as e:
        logging.error(f"Error loading counties data: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logging.error(f"Error processing study area: {e}")
        raise typer.Exit(code=1)

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    study_area_gdf.to_file(output_geojson, driver="GeoJSON")
    logging.info(f"Study area boundary saved to: {output_geojson}")
    logging.info(f"Boundary details: {len(study_area_gdf)} feature(s), CRS: {study_area_gdf.crs}")

if __name__ == "__main__":
    app() 