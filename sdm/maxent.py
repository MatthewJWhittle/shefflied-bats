import numpy as np
import geopandas as gpd
from sklearn.metrics import roc_auc_score
import elapid as ela
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray as rxr
from sdm.geo import generate_model_raster
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from elapid import MaxentModel, GeographicKFold, distance_weights


def prepare_occurence_data(
    presence_gdf: gpd.GeoDataFrame,
    background_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    input_vars: list,
    drop_na: bool = True,
    sample_weight_n_neighbors: int = 5,
):
    presence_gdf = presence_gdf.copy()
    background_gdf = background_gdf.copy()
    # Filter to the grid
    presence_gdf = filter_gdf_to_grid(presence_gdf, grid_gdf)

    # Keep only the geometry
    presence_gdf = presence_gdf[input_vars + ["geometry"]]
    background_gdf = background_gdf[input_vars + ["geometry"]]

    # Drop Missing Values
    if drop_na:
        presence_gdf.dropna(inplace=True)
        background_gdf.dropna(inplace=True)

    # Calculate sample weights
    presence_gdf["sample_weight"] = distance_weights(
        presence_gdf, n_neighbors=sample_weight_n_neighbors
    )
    background_gdf["sample_weight"] = distance_weights(
        background_gdf, n_neighbors=sample_weight_n_neighbors
    )

    occurrence = ela.stack_geodataframes(
        presence_gdf,
        background_gdf,
        add_class_label=True,
    )

    return occurrence


def filter_bats(gdf, genus=None, latin_name=None, activity_type=None):
    gdf = gdf.copy()
    if genus:
        gdf = gdf[gdf.genus == genus]
    if latin_name:
        gdf = gdf[gdf.latin_name == latin_name]
    if activity_type:
        gdf = gdf[gdf.activity_type == activity_type]
    return gdf


def extract_split(gdf: gpd.GeoDataFrame, idx: np.array) -> tuple:
    split_gdf = gdf.iloc[idx].copy()
    # Extract the training data
    X = split_gdf.drop(columns=["class", "sample_weight", "geometry"])
    y = split_gdf["class"]
    weight = split_gdf["sample_weight"]

    return X, y, weight


def cv_maxent(
    model, occurrence: gpd.GeoDataFrame, metric_fn=roc_auc_score, folds: int = 3
) -> tuple:
    gfolds = ela.GeographicKFold(n_splits=folds)
    metrics = []
    models = []

    for train_idx, test_idx in gfolds.split(occurrence):
        # this returns arrays for indexing the original dataframe
        # which requires using the pandas .iloc interface

        # Get training data
        X_train, y_train, sample_weight_train = extract_split(occurrence, train_idx)
        X_test, y_test, sample_weight_test = extract_split(occurrence, test_idx)

        # Check that y_test contains both classes
        # If not, skip this fold
        if len(np.unique(y_test)) < 2:
            continue

        # Fit the model
        model.fit(X_train, y_train, maxent__sample_weight=sample_weight_train)

        # evaluation
        ypred = model.predict(X_test)
        metric_val = metric_fn(y_test, ypred)
        metrics.append(metric_val)
        models.append(model)

    return models, np.array(metrics)

def train_maxent(
    model, occurrence: gpd.GeoDataFrame, n_jobs=1):
    """Train a maxent model on all provided data"""
    train_idx = np.arange(len(occurrence))
    # Get training data
    X_train, y_train, sample_weight_train = extract_split(occurrence, train_idx)
    # Fit the model
    model.fit(X_train, y_train, maxent__sample_weight=sample_weight_train)
    
    return model
    




# define a function to filter the gdf to keep only one point per grid index
def filter_gdf_to_grid(gdf, grid, tolerance=50):
    gdf_grid = gpd.sjoin_nearest(
        gdf,
        grid,
        how="left",
        distance_col="distance",
        max_distance=tolerance,
    )
    # Drop the duplicate records
    gdf_grid.drop_duplicates(subset="index_right", inplace=True)
    # Clean up the column names
    gdf_grid.drop(columns=["index_right", "distance"], inplace=True)
    return gdf_grid
