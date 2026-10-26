from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING

import geopandas as gpd
import numpy as np

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    """Project-level MLflow knobs; secrets live in the gitignored settings file."""

    enabled: bool = False
    settings_file: str = "mlflow.settings.yml"
    experiment_name: str = "species_sdm_experiment"
    tracking_uri: Optional[str] = None
    artifact_root: Optional[str] = None


class PredictionPublishConfig(BaseModel):
    """Opt-in versioned prediction publish settings (project-specific, not library defaults)."""

    root: Optional[str] = None


class ProjectConfig(BaseModel):
    paths: PathsConfig
    spatial: SpatialConfig
    crs: str
    mlflow: MlflowConfig
    prediction_publish: Optional[PredictionPublishConfig] = None


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


class DeepMaxentConfigModel(BaseModel):
    """Optional joint DeepMaxent (PyTorch) settings; requires ``uv sync --extra dl``."""

    device: str = "auto"  # auto | mps | cuda | cpu
    hidden_size: int = 250
    n_hidden_layers: int = 2
    dropout: float = 0.0  # 0 = off
    learning_rate: float = 2e-4
    batch_size: int = 250
    weight_decay: float = 3e-4
    epochs: int = 100
    n_seeds: int = 10
    epsilon: float = 1e-6
    # Inverse-frequency species weights (Zbinden-style); off by default — count
    # weighting can pull rare species toward common patterns.
    use_species_weights: bool = False
    transform: str = "cloglog"
    early_stopping_patience: Optional[int] = None




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


class CvConfig(BaseModel):
    """Cross-validation fold method and parameters (see ``sdm.models.maxent.folds``)."""

    method: str = "blocks"
    n_folds: int = 3
    seed: Optional[int] = None
    block_size_m: float = 10_000.0
    block_n_folds: int = 5
    block_n_tries: int = 500
    buffer_m: float = 10_000.0
    # Warn (never block) when buffered_loo presence count exceeds this (thin-species guidance).
    buffered_loo_warn_above: int = 25


class ModelConfig(BaseModel):
    seed: int
    # Trainer: ``maxent`` (elapid, default) or experimental ``deepmaxent`` (joint MLP).
    type: str = "maxent"
    maxent: MaxentConfigModel
    deepmaxent: Optional[DeepMaxentConfigModel] = None
    sampling: Optional[SamplingConfig] = None
    background: Optional[BackgroundConfig] = None
    cv: Optional[CvConfig] = None
    # When true, publish no cut-off (no threshold value / thresholded map).
    model_limited: bool = False

    @field_validator("type", mode="before")
    @classmethod
    def validate_model_type(cls, v: Any) -> str:
        name = str(v or "maxent").strip().lower()
        if name not in {"maxent", "deepmaxent"}:
            raise ValueError(
                f"model.type must be 'maxent' or 'deepmaxent', got {v!r}"
            )
        return name



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
    model_limited: bool = False


class TrainingResults(SDMModel):
    """Results from training a single model."""

    final_model: Optional[object] = None
    cv_models: Optional[List[object]] = None
    cv_scores: Optional[np.ndarray] = None
    validation_scores: Optional[Any] = None
    cv_n_folds: Optional[int] = None
    fold_method: Optional[str] = None
    fold_params: Optional[Dict[str, Any]] = None
    fold_presence_counts: Optional[List[int]] = None
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



