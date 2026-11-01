"""Build joint multi-output training tables from per-model TrainingData."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sdm.commands.modelling.utils import get_model_id
from sdm.types import TrainingData

logger = logging.getLogger(__name__)


@dataclass
class JointDeepMaxentTable:
    """Sites × features with a multi-label presence matrix for all outputs."""

    features: pd.DataFrame
    targets: pd.DataFrame  # columns = model_ids, values = 0/1 (or counts)
    geometry: gpd.GeoSeries
    feature_names: List[str]
    output_names: List[str]
    model_limited: Dict[str, bool]

    def as_geodataframe(self) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(
            pd.concat([self.features, self.targets], axis=1),
            geometry=self.geometry,
            crs=self.geometry.crs,
        )
        return gdf


def _model_key(data: TrainingData) -> str:
    return get_model_id([data.latin_name, data.activity_type])


def shared_feature_names(training_data: Sequence[TrainingData]) -> List[str]:
    """Union of per-model feature lists (sorted for stability).

    A shared trunk needs one feature roster. Per-model MaxEnt subsets still
    apply to the elapid baseline; DeepMaxent uses this union so every output
    head sees the same environmental representation.
    """
    names: set[str] = set()
    for data in training_data:
        names.update(data.model_features)
    if not names:
        raise ValueError("No model features found in training_data")
    return sorted(names)


def build_joint_presence_background_table(
    training_data: Sequence[TrainingData],
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> JointDeepMaxentTable:
    """Stack per-model presence / background into one multi-output table.

    Presence rows keep a 1 in their model column. Background rows (class 0)
    from every model are kept as shared all-bat / TGB sites with zeros across
    outputs; duplicate geometries are aggregated (max presence flag per cell).
    Coordinates are never used as model inputs — only for aggregation / CV.
    """
    if not training_data:
        raise ValueError("training_data is empty")

    feats = list(feature_names) if feature_names is not None else shared_feature_names(
        training_data
    )
    output_names = sorted(_model_key(d) for d in training_data)
    model_limited = {_model_key(d): bool(d.model_limited) for d in training_data}

    rows: List[pd.DataFrame] = []
    for data in training_data:
        key = _model_key(data)
        occ = data.occurrence.copy()
        if "class" not in occ.columns:
            raise ValueError(f"{key}: occurrence missing 'class' column")
        missing = [c for c in feats if c not in occ.columns]
        if missing:
            raise ValueError(f"{key}: missing features for joint table: {missing}")

        frame = occ[feats].copy()
        frame["geometry"] = list(occ.geometry.values)
        for name in output_names:
            frame[name] = 0.0
        presence = occ["class"].to_numpy() == 1
        frame.loc[presence, key] = 1.0
        # Keep presence rows and background rows (class 0) as DeepMaxent sites.
        rows.append(frame)

    stacked = pd.concat(rows, ignore_index=True)
    geom = gpd.GeoSeries(stacked["geometry"], crs=training_data[0].occurrence.crs)
    # Aggregate duplicate sites by rounded coordinates (100 m grid friendly).
    stacked["_x"] = geom.x.round(1).to_numpy()
    stacked["_y"] = geom.y.round(1).to_numpy()
    agg_feats = {c: "mean" for c in feats}
    agg_out = {c: "max" for c in output_names}
    grouped = (
        stacked.groupby(["_x", "_y"], as_index=False)
        .agg({**agg_feats, **agg_out, "geometry": "first"})
    )
    geometry = gpd.GeoSeries(grouped["geometry"], crs=training_data[0].occurrence.crs)
    features = grouped[feats].astype(np.float64)
    targets = grouped[output_names].astype(np.float64)
    # Drop rows with any NaN features (same rule as MaxEnt EV drop).
    valid = features.notna().all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        logger.info("DeepMaxent joint table dropped %d sites with NA features", n_drop)
    features = features.loc[valid].reset_index(drop=True)
    targets = targets.loc[valid].reset_index(drop=True)
    geometry = geometry.loc[valid].reset_index(drop=True)

    logger.info(
        "DeepMaxent joint table: %d sites × %d features × %d outputs",
        len(features),
        len(feats),
        len(output_names),
    )
    return JointDeepMaxentTable(
        features=features,
        targets=targets,
        geometry=geometry,
        feature_names=feats,
        output_names=output_names,
        model_limited=model_limited,
    )


def species_presence_background_labels(
    targets: pd.DataFrame,
    output_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary class labels for one output: presence (1) vs other sites (0).

    Other sites act as the background / TGB comparison set for that head when
    scoring held-out AUC (comparable to MaxEnt presence-background AUC).
    """
    y = targets[output_name].to_numpy(dtype=float)
    classes = (y > 0).astype(int)
    return classes, y
