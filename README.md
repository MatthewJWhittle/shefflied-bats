

# Modelling steps

1. Generate the study area
2. Download the datasets (see Prepare EVs)
3. Prepare the datasets (see Prepare EVs) (run `data_prep/generate_evs/main.py`)
4. Work through the `data_prep/prep_model_data/feature-engineering.ipynb` notebook to generate the features for the model. This will output a file called `ev_clusters.csv` which you annotate a copy of (`ev_clusters_selected.csv`) with the features you want to use in the model. This will be used in the next step.
5. Prepare the occurrence data (see Prepare Occurence Data) & save as `data/processed/bats-tidy.geojson`
6. Generate the background points using:
```bash
python data_prep/generate_background_points.py --occurrence_path data/processed/bats-tidy.geojson --boundary data/processed/boundary.geojson --output data/processed
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

