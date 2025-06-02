# Sheffield Bats Species Distribution Modelling

This project implements Species Distribution Modelling (SDM) for bat species in Sheffield, UK. It processes environmental variables, occurrence data, and generates predictive models to understand bat species distribution patterns.

## Quick Start

1. **Prerequisites**
   - Python 3.11 or higher
   - Git
   - Sufficient disk space (~20GB for data and models)
   - Access to external data sources (OS, CEH, BGS)

2. **Installation**
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Clone and setup
   git clone [repository-url]
   cd sheffield-bats
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Download Required Data**
   - OS Vector Map District tiles
   - CEH Land Cover data
   - BGS GeoCoast data
   See [Required Data Sources](#required-data-sources) for details.

4. **Basic Workflow**
   ```bash
   # 1. Define study area
   sdm boundary
   
   # 2. Process environmental variables
   sdm terrain
   sdm landcover
   sdm coastal
   sdm os
   
   # 3. Merge variables
   sdm merge --dataset-inputs "terrain=data/evs/terrain/terrain.tif" --dataset-inputs "landcover=data/evs/landcover/landcover.tif"
   
   # 4. Generate background points
   sdm background --occurrence-data-path data/processed/bats-tidy.geojson --boundary-path data/processed/boundary.geojson
   
   # 5. Train models
   sdm train --bats-file data/processed/bats-tidy.geojson --background-file data/processed/background-points.geojson --ev-file data/evs/evs-to-model.tif
   
   # 6. Generate predictions
   sdm predict --ev-path data/evs/evs-to-model.tif --models-dir data/sdm_models
   ```

## Key Features

- **Environmental Variables**: Processing of terrain, land cover, coastal distance, and OS data
- **Model Training**: Species distribution modeling with configurable parameters
- **Visualization**: Generation of model outputs and partial dependence plots
- **CLI Interface**: Easy-to-use command-line tools for all operations
- **Reproducibility**: Configuration-driven processing with MLflow experiment tracking

## Configuration

The project uses YAML configuration files in the `config/` directory:
- `default.yaml`: Main configuration file
- `spatial.json`: Spatial reference settings
- `input_variables.json`: Environmental variable definitions

Key configuration settings:
- Coordinate Reference System: OSGB36 (EPSG:27700)
- MLflow tracking for experiment management
- Model hyperparameters

## Dependencies

Key Python packages:
- `geopandas`: Spatial data processing
- `rasterio`: Raster data handling
- `scikit-learn`: Machine learning models
- `mlflow`: Experiment tracking
- `elapid`: Species distribution modeling

See `pyproject.toml` for complete dependency list.

## Overview

The project uses machine learning to model bat species distributions based on environmental variables and occurrence data. It includes data preparation pipelines, model training, and visualization tools.

## Project Structure

```
<root>/
├── config/                   # Configuration files
├── data/                     # Data directory
│   ├── raw/                 # Raw input data
│   └── processed/           # Processed data files
├── sdm/                     # Core library
│   ├── commands/           # CLI command implementations
│   ├── data/               # Data processing modules
│   ├── models/             # Model implementations
│   ├── raster/             # Raster processing utilities
│   ├── utils/              # Utility functions
│   └── viz/                # Visualization tools
├── notebooks/               # Jupyter notebooks
├── tests/                   # Test suite
└── outputs/                 # Model outputs and results
```

## Data Directory Structure

The project uses a structured data directory to manage different stages of data processing:

```
data/
├── raw/                     # Raw input data
│   ├── bats/               # Original bat observation data
│   └── big-files/          # Large external datasets
│       ├── BGS GeoCoast/   # Coastal data from BGS
│       ├── CEH/            # Land cover data
│       └── os-vector-map/  # OS Vector Map District tiles
│
├── processed/              # Processed and intermediate data
│   ├── bats-tidy.geojson  # Cleaned and standardized bat observations
│   ├── boundary.geojson   # Study area boundary
│   ├── background-points.geojson  # Generated background points
│   └── bat_density.tif    # Bat occurrence density raster
│
├── evs/                    # Environmental variables
│   ├── evs-to-model.tif   # Final combined environmental variables
│   ├── terrain_stats.tif  # Terrain-derived variables
│   ├── coastal_distance.tif  # Distance to coast
│   ├── ceh-land-cover-100m.tif  # Land cover data
│   ├── climate_stats.tif  # Climate variables
│   └── grid-points.parquet  # Sampling grid points
│
├── sdm_models/            # Trained species distribution models
├── sdm_predictions/       # Model prediction outputs
└── temp/                  # Temporary processing files
```

### Key Data Files

1. **Raw Data** (`data/raw/`)
   - Original bat observation data
   - External datasets (OS, CEH, BGS) that need to be downloaded

2. **Processed Data** (`data/processed/`)
   - `bats-tidy.geojson`: Standardized bat observations with required fields
   - `boundary.geojson`: Study area boundary
   - `background-points.geojson`: Generated background points for modeling
   - `bat_density.tif`: Raster of bat occurrence density

3. **Environmental Variables** (`data/evs/`)
   - `evs-to-model.tif`: Combined environmental variables for modeling
   - `terrain_stats.tif`: Terrain-derived variables (slope, aspect, etc.)
   - `coastal_distance.tif`: Distance to coast
   - `ceh-land-cover-100m.tif`: Land cover classification
   - `climate_stats.tif`: Climate variables (temperature, precipitation)
   - `grid-points.parquet`: Sampling grid for model training

4. **Model Outputs**
   - `sdm_models/`: Trained species distribution models
   - `sdm_predictions/`: Model prediction rasters
   - `outputs/`: Additional model outputs and visualizations

## Usage

The project provides a comprehensive command-line interface (CLI) for all operations. For detailed documentation of all available commands and their options, please refer to the [CLI Documentation](sdm/README.md).

### Basic Workflow

1. **Define Study Area**
   ```bash
   sdm boundary
   ```

2. **Process Environmental Variables**
   ```bash
   # Generate terrain data
   sdm terrain
   
   # Process land cover data
   sdm landcover
   
   # Generate coastal distance
   sdm coastal
   
   # Process OS data
   sdm os
   
   # Merge all environmental variables
   sdm merge --dataset-inputs "terrain=data/evs/terrain/terrain.tif" --dataset-inputs "landcover=data/evs/landcover/landcover.tif"
   ```

3. **Generate Background Points**
   ```bash
   sdm background --occurrence-data-path data/processed/bats-tidy.geojson --boundary-path data/processed/boundary.geojson
   ```

4. **Train Models**
   ```bash
   sdm train --bats-file data/processed/bats-tidy.geojson --background-file data/processed/background-points.geojson --ev-file data/evs/evs-to-model.tif
   ```

5. **Generate Predictions**
   ```bash
   sdm predict --ev-path data/evs/evs-to-model.tif --models-dir data/sdm_models
   ```

6. **Create Visualizations**
   ```bash
   sdm visualize --run-summary-path outputs/sdm_runs/sdm_run_summary.csv --ev-raster-path data/evs/evs-to-model.tif
   ```

For more detailed information about each command, including all available options and configuration settings, please refer to the [CLI Documentation](sdm/README.md).

## Required Data Sources

### OS Data
Download Vector Map District tiles from [OS Data Products](https://www.ordnancesurvey.co.uk/products/os-vectormap-district)
- Save to: `data/raw/big-files/os-vector-map`

### Land Cover Data
Download from [Land Cover Map](https://www.ceh.ac.uk/data/ukceh-land-cover-maps)
- Save to: `data/raw/big-files/CEH`

### BGS GeoCoast
Download from [BGS website](https://www.bgs.ac.uk/download/bgs-geocoast-open/)
- Save to: `data/raw/big-files/BGS GeoCoast`

## Contributing

[Add contribution guidelines]

## License

[Add license information]

## Contact

[Add contact information]

# Modelling steps

1. Generate the study area
2. Download the datasets (see Prepare EVs)
3. Prepare the datasets (see Prepare EVs) (run `data_prep/generate_evs/main.py`)
4. Work through the `data_prep/prep_model_data/feature-engineering.ipynb` notebook to generate the features for the model. This will output a file called `ev_clusters.csv` which you annotate a copy of (`ev_clusters_selected.csv`) with the features you want to use in the model. This will be used in the next step.
5. Prepare the occurrence data (see Prepare Occurence Data) & save as `data/processed/bats-tidy.geojson`
6. Generate the background points using:
```bash
uv run -m data_prep.prep_model_data.background_points --occurrence_path data/processed/bats-tidy.geojson --boundary data/processed/boundary.geojson --output data/processed --n-points 4000 
```
Which will create a file called `data/processed/background-points.geojson`. 

## Define the Study Area
Update the logic in `data_prep/generate_evs/ingestion/load_study_area.py` to create a study area for the model. The study area should be a geodataframe.

This file will be saved to `data/processed/boundary.geojson` and used in subsequent steps.

You need to download the boundary data from the [ONs Boundaries data](https://geoportal.statistics.gov.uk/search?q=BDY_CTYUA%202024&sort=Title%7Ctitle%7Casc) and save it as:

`data/raw/big-files/Counties_and_Unitary_Authorities_May_2023_UK_BFC_7858717830545248014.geojson`

## Prepare EVs

The logic for preparing environmental variables can be executed by running `data_prep/generate_evs/main.py`. This script will acquire each data source and prepare it for the model, then finally combine them into one GeoTiff.

Most data sources are pulled directly from their live servers but others need to be downloaded and saved locally. These include:

### OS Data
A folder of OS Vector Map District tiles downloaded from the [OS Data Products site](https://www.ordnancesurvey.co.uk/products/os-vectormap-district)

Download each tile in your study area and save them in the following directory:
`data/raw/big-files/os-vector-map` 

### Land Cover Data
A folder of Land Cover data downloaded from the [Land Cover Map](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) site.

Downloaded to here:
`data/raw/big-files/CEH`


### BGS GeoCoast

The [BGS GeoCoast data is available from the BGS website](https://www.bgs.ac.uk/download/bgs-geocoast-open/). You need to download the data and save it in the following directory:

`data/raw/big-files/BGS GeoCoast`

## Prepare Occurence Data

For this step you need to transform your occurence data into one geojson file. You can do this however you likem, there is no pipeline since this is very dataset specific.

Save the file at `data/processed/bats-tidy.geojson`

The file should have the following columns:

- `unique_id` (`string`): A unique identifier for the observation. This can be a combination of the species name and the date of the observation.
- `latin_name` (`string`): The latin name of the species. This should be in the format `Genus species` (e.g. `Pipistrellus pipistrellus`).
- `roosting` (`string`): A string indicating whether the observation was made while the bat was in flight or roosting. This should be one of the following values: `In Flight`, `Roosting`.
- `date` (`string`): The date of the observation in the format `YYYY-MM-DD`. This should be the date of the observation, not the date of the file.
- `x` (`float`): The x coordinate of the observation in British National Grid (OSGB36) coordinates (27700).
- `y` (`float`): The y coordinate of the observation in British National Grid (OSGB36) coordinates (27700).
- `accuracy` (`float`): The accuracy of the observation in meters. Should be inferred from the resolution of the alpha numeric grid reference if that is the source geometry. (float)
- `geometry` (`Point`): A shapely Point object representing the location of the observation. This should be in the format `Point(x, y)`.

