from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING

import geopandas as gpd
import numpy as np

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from sdm.occurrence.sampling import BackgroundMethod, TransformMethod

class PathsConfig(BaseModel):
    raw_data: str
    processed_data: str
    models: str
    predictions: str
    model_config_path: str = Field(alias="model_config")
    variables_config_path: str = Field(alias="variables_config")
    tuning_dir: str
    config_dir: str
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
    background_method: Any = Field(default="contrast")  # type: ignore[assignment]
    transform_method: Any = Field(default="presence")  # type: ignore[assignment]
    
    @field_validator("background_method", mode="before")
    @classmethod
    def validate_background_method(cls, v: Any) -> Any:
        """Normalize and convert background_method string to enum."""
        # Lazy import to avoid circular dependency
        from sdm.occurrence.sampling import BackgroundMethod
        
        if isinstance(v, BackgroundMethod):
            return v
        if isinstance(v, str):
            return BackgroundMethod(v.lower())
        return v
    
    @field_validator("transform_method", mode="before")
    @classmethod
    def validate_transform_method(cls, v: Any) -> Any:
        """Normalize and convert transform_method string to enum."""
        # Lazy import to avoid circular dependency
        from sdm.occurrence.sampling import TransformMethod
        
        if isinstance(v, TransformMethod):
            return v
        if isinstance(v, str):
            return TransformMethod(v.lower())
        return v
    
    model_config = ConfigDict(populate_by_name=True)


class SamplingConfig(BaseModel):
    min_presence: int = 15
    subset_occurrence: Optional[int] = None
    subset_background: bool = True
    order_by_density_for_subset: bool = True
    sample_weight_n_neighbors: int = 5
    background: SamplingBackgroundConfig = SamplingBackgroundConfig()
    grid_size_m: float = 2000


class ModelConfig(BaseModel):
    seed: int
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
    feature_caps: Dict[str, float] = Field(default_factory=dict)  # Resolved upper caps, applied in the pipeline


class TrainingResults(SDMModel):
    """Results from training a single model."""

    final_model: Optional[object] = None
    cv_models: Optional[List[object]] = None
    cv_scores: Optional[np.ndarray] = None
    validation_scores: Optional[Any] = None
    cv_n_folds: Optional[int] = None
    success: bool = False
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FeatureCapConfig(BaseModel):
    """Upper cap for one predictor: a quantile of the training points or a fixed value."""

    quantile: Optional[float] = None
    value: Optional[float] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "FeatureCapConfig":
        if (self.quantile is None) == (self.value is None):
            raise ValueError("A feature cap needs exactly one of 'quantile' or 'value'.")
        if self.quantile is not None and not 0.0 < self.quantile <= 1.0:
            raise ValueError("Cap 'quantile' must be in (0, 1].")
        return self


class VariablesConfig(BaseModel):
    variables: List[str]
    caps: Dict[str, FeatureCapConfig] = Field(default_factory=dict)

    def validate_features(self, features: List[str]) -> "VariablesConfig":
        """Validate that all variables in this config exist in the provided features list."""
        if not set(self.variables).issubset(set(features)):
            raise ValueError(f"Variables config contains features that are not in the features list: {set(self.variables) - set(features)}")
        uncapped = set(self.caps) - set(self.variables)
        if uncapped:
            raise ValueError(f"Caps set for variables not in the variables list: {sorted(uncapped)}")
        return self



