from typing import List, Optional, Any

import geopandas as gpd
import numpy as np

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sdm.occurrence.sampling import BackgroundMethod, TransformMethod

class PathsConfig(BaseModel):
    raw_data: str
    processed_data: str
    models: str
    predictions: str
    model_config_path: str = Field(alias="model_config")
    variables_config_path: str = Field(alias="variables_config")
    tuning_dir: str
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


class BackgroundConfig(BaseModel):
    """Configuration for background point generation with typed enum properties."""
    
    n_background_points: int = 4000
    background_value: float = 0.00
    sigma: float = 6.5
    background_method: BackgroundMethod = Field(default=BackgroundMethod.CONTRAST)
    transform_method: TransformMethod = Field(default=TransformMethod.PRESENCE)
    
    @field_validator("background_method", "transform_method", mode="before")
    @classmethod
    def normalize_enum(cls, v: Any) -> str:
        """Normalize enum input to lowercase string for StrEnum conversion."""
        if isinstance(v, (BackgroundMethod, TransformMethod)):
            return v.value
        if isinstance(v, str):
            return v.lower()
        return v
    
    model_config = ConfigDict(populate_by_name=True)


class SamplingConfig(BaseModel):
    min_presence: int = 15
    subset_occurrence: Optional[int] = None
    subset_background: bool = True
    order_by_density_for_subset: bool = True
    sample_weight_n_neighbors: int = 5
    background: SamplingBackgroundConfig = SamplingBackgroundConfig()


class ModelConfig(BaseModel):
    maxent: MaxentConfigModel
    sampling: Optional[SamplingConfig] = None
    background: Optional[BackgroundConfig] = None


class SDMModel(BaseModel):
    latin_name: str
    activity_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def identifier(self) -> str:
        return f"{self.latin_name}_{self.activity_type}"


class TrainingData(SDMModel):
    occurrence: gpd.GeoDataFrame
    maxent_config: Any  # DefaultMaxentConfig - using Any to avoid circular imports, required for training
    model_features: List[str]  # Required for training - list of feature column names to use


class TrainingResults(SDMModel):
    """Results from training a single model."""

    final_model: Optional[object] = None
    cv_models: Optional[List[object]] = None
    cv_scores: Optional[np.ndarray] = None
    success: bool = False
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VariablesConfig(BaseModel):
    variables: List[str]

    def validate_features(self, features: List[str]) -> "VariablesConfig":
        """Validate that all variables in this config exist in the provided features list."""
        if not set(self.variables).issubset(set(features)):
            raise ValueError(f"Variables config contains features that are not in the features list: {set(self.variables) - set(features)}")
        return self

