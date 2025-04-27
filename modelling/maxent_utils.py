import warnings
from typing import Optional
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.metrics import roc_auc_score
import elapid as ela
from elapid import distance_weights



import rasterio as rio
from sklearn.base import BaseEstimator

from elapid.types import to_iterable
from elapid.utils import (
    NoDataException,
    check_raster_alignment,
    create_output_raster_profile,
    get_raster_band_indexes,
    tqdm_opts,
    get_tqdm,
)

from elapid.geo import (
    apply_model_to_array
)


tqdm = get_tqdm()


def calculate_background_points(n_presences, min_bg=1000, max_bg=10000, factor=10):
    """
    Calculate a recommended number of background (pseudo-absence) points
    based on the number of presence records.
    
    Args:
        n_presences (int): Number of presence records.
        min_bg (int): Minimum number of background points (default 1,000).
        max_bg (int): Maximum number of background points (default 10,000).
        factor (int): Multiplier factor for presences (default 10).
        
    Returns:
        int: Calculated number of background points.
    """
    # Scale by factor, then cap between min_bg and max_bg
    n_bg = max(min(n_presences * factor, max_bg), min_bg)
    return int(n_bg)

def prepare_occurence_data(
    presence_gdf: gpd.GeoDataFrame,
    background_gdf: gpd.GeoDataFrame,
    background_density: pd.Series,
    grid_gdf: gpd.GeoDataFrame,
    input_vars: list,
    drop_na: bool = True,
    sample_weight_n_neighbors: int = 5,
    filter_to_grid: bool = True,
    subset_background: bool = True,
):
    presence_gdf = presence_gdf.copy() # type: ignore
    background_gdf = background_gdf.copy() # type: ignore
    # Filter to the grid
    if filter_to_grid:
        
        # TODO: Improve this grid filtering - dropping too many points
        # Also not removing conficts between presence and background (
        # .eg squares where both presence and background points are present)
        presence_gdf = filter_gdf_to_grid(presence_gdf, grid_gdf)
        background_gdf = filter_gdf_to_grid(background_gdf, grid_gdf)

        # Drop background points that have a grid_index in the presence points
        background_gdf = background_gdf[
            ~background_gdf["grid_index"].isin(presence_gdf["grid_index"])
        ]
        # filter the density to gdf index
        background_density = background_density.loc[
            background_gdf.index
        ]


        # Drop the grid index column
        presence_gdf.drop(columns=["grid_index"], inplace=True)
        background_gdf.drop(columns=["grid_index"], inplace=True)

    if subset_background:
        # Subset the background points to the number of presence points
        n_bg = calculate_background_points(len(presence_gdf))
        
        # order the background points from highest to lowest density
        # then take the top n_bg points
        background_gdf = background_gdf.loc[
            background_density.sort_values(ascending=False).index[:n_bg]
        ]
        

    # Keep only the geometry
    presence_gdf = presence_gdf[input_vars + ["geometry"]] # type: ignore
    background_gdf = background_gdf[input_vars + ["geometry"]] # type: ignore

    # Drop Missing Values
    if drop_na:
        presence_gdf.dropna(inplace=True)
        background_gdf.dropna(inplace=True)

    occurrence = ela.stack_geodataframes(
        presence_gdf,
        background_gdf,
        add_class_label=True,
    )
    occurrence["sample_weight"] = distance_weights(
        occurrence, n_neighbors=sample_weight_n_neighbors
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


def extract_split(gdf: gpd.GeoDataFrame, idx: np.ndarray) -> tuple:
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
    # Rename to grid index 
    gdf_grid.rename(columns={"index_right": "grid_index"}, inplace=True)
    # Clean up the column names
    gdf_grid.drop(columns=["distance"], inplace=True)
    return gdf_grid



from sklearn.base import clone

def eval_train_model(occurrence, model):
    cv_models, cv_scores = cv_maxent(
        model = clone(model),
        occurrence = occurrence
    )
        # Train a model on all the data
    full_model = train_maxent(
        model=clone(model),
        occurrence=occurrence,
    )
    return full_model, cv_models, cv_scores

# Impliment a custom scaler that adds a small epsilon to the data to avoid zero values
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer


def add_epsilon(X, epsilon=1e-5):
    X = X.copy()
    X += epsilon
    return X

class ZeroDivScaler(FunctionTransformer):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        super().__init__(func=add_epsilon, kw_args={"epsilon": epsilon})





def apply_model_to_rasters(
    model: BaseEstimator,
    raster_paths: list,
    output_path: str,
    resampling: rio.enums.Enum = rio.enums.Resampling.average,
    count: int = 1,
    dtype: str = "float32",
    nodata: float = -9999,
    driver: str = "GTiff",
    compress: str = "deflate",
    bigtiff: bool = True,
    template_idx: int = 0,
    windowed: bool = True,
    predict_proba: bool = False,
    ignore_sklearn: bool = True,
    quiet: bool = False,
    **kwargs,
) -> None:
    """Applies a trained model to a list of raster datasets.

    The list and band order of the rasters must match the order of the covariates
    used to train the model. It reads each dataset block-by-block, applies
    the model, and writes gridded predictions. If the raster datasets are not
    consistent (different extents, resolutions, etc.), it wll re-project the data
    on the fly, with the grid size, extent and projection based on a 'template'
    raster.

    Args:
        model: object with a model.predict() function
        raster_paths: raster paths of covariates to apply the model to
        output_path: path to the output file to create
        resampling: resampling algorithm to apply to on-the-fly reprojection
            from rasterio.enums.Resampling
        count: number of bands in the prediction output
        dtype: the output raster data type
        nodata: output nodata value
        driver: output raster format
            from rasterio.drivers.raster_driver_extensions()
        compress: compression to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        template_idx: index of the raster file to use as a template.
            template_idx=0 sets the first raster as template
        windowed: apply the model using windowed read/write
            slower, but more memory efficient
        predict_proba: use model.predict_proba() instead of model.predict()
        ignore_sklearn: silence sklearn warning messages
        quiet: silence progress bar output
        **kwargs: additonal keywords to pass to model.predict()

    Returns:
        None: saves model predictions to disk.
    """
    # make sure the raster_paths are iterable
    raster_paths = to_iterable(raster_paths)

    # get and set template parameters
    windows, dst_profile = create_output_raster_profile(
        raster_paths,
        template_idx,
        count=count,
        windowed=windowed,
        nodata=nodata,
        compress=compress,
        driver=driver,
        bigtiff=bigtiff,
    )

    # get the bands and indexes for each covariate raster
    nbands, band_idx = get_raster_band_indexes(raster_paths)

    # check whether the raster paths are aligned to determine how the data are read
    aligned = check_raster_alignment(raster_paths)

    # set a dummy nodata variable if none is set
    # (acutal nodata reads handled by rasterios src.read(masked=True) method)
    nodata = nodata or 0

    # turn off sklearn warnings
    if ignore_sklearn:
        warnings.filterwarnings("ignore", category=UserWarning)

    # open all rasters to read from later
    srcs = [rio.open(raster_path) for raster_path in raster_paths]

    # use warped VRT reads to align all rasters pixel-pixel if not aligned
    if not aligned:
        vrt_options = {
            "resampling": resampling,
            "transform": dst_profile["transform"],
            "crs": dst_profile["crs"],
            "height": dst_profile["height"],
            "width": dst_profile["width"],
        }
        srcs = [rio.vrt.WarpedVRT(src, **vrt_options) for src in srcs]

    # read and reproject blocks from each data source and write predictions to disk
    with rio.open(output_path, "w", **dst_profile) as dst:
        for window in tqdm(windows, desc="Window", disable=quiet, **tqdm_opts):
            # create stacked arrays to handle multi-raster, multi-band inputs
            # that may have different nodata locations
            covariates = np.zeros((nbands, window.height, window.width), dtype=np.float32)
            nodata_idx = np.ones_like(covariates, dtype=bool)

            try:
                for i, src in enumerate(srcs):
                    data = src.read(window=window, masked=True)
                    covariates[band_idx[i] : band_idx[i + 1]] = data
                    nodata_idx[band_idx[i] : band_idx[i + 1]] = data.mask

                    # skip blocks full of no-data
                    if nodata_idx.any(axis=0).all():
                        raise NoDataException()

                predictions = apply_model_to_array(
                    model,
                    covariates,
                    nodata,
                    nodata_idx,
                    count=count,
                    dtype=dtype,
                    predict_proba=predict_proba,
                    **kwargs,
                )
                dst.write(predictions, window=window)

            except NoDataException:
                continue
