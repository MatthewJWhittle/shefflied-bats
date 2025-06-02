# Species Distribution Modelling CLI

A command-line interface for processing and modeling species distribution data. This tool provides a comprehensive set of commands for environmental variable processing, model training, and visualization.

## Installation

The CLI is part of the `sdm` package. Install it using pip:

```bash
pip install -e .
```

## Available Commands

### Data Preparation

#### Boundary Creation
```bash
sdm boundary [OPTIONS]
```
Creates the study area boundary GeoJSON file by loading UK counties, filtering for a defined Yorkshire region, reprojecting, simplifying, and dissolving.

#### Background Points Generation
```bash
sdm background [OPTIONS]
```
Generates background points for species distribution modeling based on density-smoothed occurrence data.

#### Environmental Variables

##### Terrain Data
```bash
sdm terrain [OPTIONS]
```
Downloads and processes DTM and DSM terrain data from EA WCS services.

##### Climate Data
```bash
sdm climate [OPTIONS]
```
Downloads and processes climate data for the study area.

##### Land Cover Data
```bash
sdm landcover [OPTIONS]
```
Processes CEH land cover data based on a given boundary.

##### Vegetation Object Model (VOM)
```bash
sdm vom [OPTIONS]
```
Downloads Vegetation Object Model (VOM) data and summarizes it to various metrics.

##### Terrain Statistics
```bash
sdm terrain-stats [OPTIONS]
```
Calculates various terrain statistics from an input Digital Elevation Model (DEM).

##### Coastal Distance
```bash
sdm coastal [OPTIONS]
```
Generates a coastal distance raster layer using BGS GeoCoast data.

##### OS Data Processing
```bash
sdm os [OPTIONS]
```
Processes Ordnance Survey data to generate environmental variables.

##### Merge Environmental Layers
```bash
sdm merge [OPTIONS]
```
Merges multiple raster datasets into a single multi-band GeoTIFF file.

### Modeling

#### Feature Extraction
```bash
sdm extract [OPTIONS]
```
Extracts environmental variable data for occurrence/background points and saves it as model-ready feature data.

#### Model Training
```bash
sdm train [OPTIONS]
```
Runs the MaxEnt model training pipeline for species distribution modeling.

#### Model Prediction
```bash
sdm predict [OPTIONS]
```
Generates predictions using trained SDM models.

### Visualization

#### Model Outputs
```bash
sdm visualize [OPTIONS]
```
Generates visualizations (e.g., Partial Dependence Plots) for trained SDM models.

## Common Options

Most commands support these common options:

- `--verbose`, `-v`: Enable verbose logging
- `--help`: Show help message and exit

## Example Workflow

1. Create study boundary:
```bash
sdm boundary --output-geojson data/processed/boundary.geojson
```

2. Generate background points:
```bash
sdm background --occurrence-data-path data/raw/bats.geojson --boundary-path data/processed/boundary.geojson
```

3. Process environmental variables:
```bash
sdm terrain --output-dir data/evs/terrain
sdm climate --output-dir data/evs/climate
sdm landcover --output-dir data/evs/landcover
```

4. Merge environmental layers:
```bash
sdm merge --dataset-inputs "terrain=data/evs/terrain/terrain.tif" --dataset-inputs "climate=data/evs/climate/climate.tif"
```

6. Train models:
```bash
sdm train --bats-file data/processed/bats-tidy.geojson --background-file data/processed/background-points.geojson
```

7. Generate predictions:
```bash
sdm predict --ev-path data/evs/evs-to-model.tif --models-dir data/sdm_models
```

8. Create visualizations:
```bash
sdm visualize --run-summary-path outputs/sdm_runs/sdm_run_summary.csv
```

## Configuration

The CLI uses default paths for input and output files, but these can be overridden using command-line options. Common default paths include:

- Boundary: `data/processed/boundary.geojson`
- Environmental variables: `data/evs/`
- Model outputs: `data/sdm_models/`
- Predictions: `data/sdm_predictions/`
- Visualizations: `outputs/sdm_visualisations/`

## Contributing

To add new commands or modify existing ones:

1. Add your command function to the appropriate module in `sdm/commands/`
2. Register the command in `sdm/cli.py`
3. Update this README with documentation for your new command

## License

This project is licensed under the MIT License - see the LICENSE file for details. 