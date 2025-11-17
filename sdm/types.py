from typing import List, Optional, Dict

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    raw_data: str
    processed_data: str
    models: str
    predictions: str
    model_config_path: str = Field(alias="model_config")
    variables_config_path: str = Field(alias="variables_config")
    occurence_data: str
    background_points: str
    boundary: str
    grid_points: str
    evs: str
    ev_tiff: str
    model_config = ConfigDict(populate_by_name=True)


class SpatialConfig(BaseModel):
    top: float
    left: float
    crs: str
    resolution: int
    study_area_buffer: float


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class ProjectConfig(BaseModel):
    paths: PathsConfig
    spatial: SpatialConfig
    crs: str
    mlflow: MlflowConfig


class MaxentConfigModel(BaseModel):
    feature_types: List[str]
    beta_multiplier: float
    beta_lqp: float
    beta_hinge: float
    beta_threshold: float
    beta_categorical: float
    n_hinge_features: int
    n_threshold_features: int
    clamp: bool
    convergence_tolerance: float
    use_lambdas: str
    n_lambdas: int
    class_weights: str | float
    tau: float
    transform: str


class SamplingBackgroundConfig(BaseModel):
    factor: int = 10
    min_bg: int = 1000
    max_bg: int = 10000


class SamplingConfig(BaseModel):
    min_presence: int = 15
    subset_occurrence: Optional[int] = None
    subset_background: bool = True
    order_by_density_for_subset: bool = True
    sample_weight_n_neighbors: int = 5
    background: SamplingBackgroundConfig = SamplingBackgroundConfig()


class ModelConfig(BaseModel):
    record_age_years: int
    maxent: MaxentConfigModel
    sampling: SamplingConfig


class SDMModel(BaseModel):
    latin_name: str
    activity_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def identifier(self) -> str:
        return f"{self.latin_name}_{self.activity_type}"


class TrainingData(SDMModel):
    occurrence: gpd.GeoDataFrame


class TrainingResults(SDMModel):
    """Results from training a single model."""

    final_model: Optional[object] = None
    cv_models: Optional[List[object]] = None
    cv_scores: Optional[np.ndarray] = None
    success: bool = False
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VariablesConfig(BaseModel):
    roster: List[str]
    activity_feature_sets: Dict[str, List[str]]


