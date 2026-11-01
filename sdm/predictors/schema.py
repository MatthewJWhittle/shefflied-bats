"""Pydantic schema for predictor-layer documentation YAML."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LayerKind(str, Enum):
    """Whether the layer is a pipeline source product or a derived transform."""

    SOURCE = "source"
    DERIVED = "derived"


class SourceInfo(BaseModel):
    """Public metadata for an upstream dataset (also the body of a source doc)."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    product: str
    licence: str
    licence_url: Optional[str] = None
    access_url: Optional[str] = None
    citation: Optional[str] = None
    version: Optional[str] = None
    notes: Optional[str] = None


class SourceDoc(BaseModel):
    """Machine-readable upstream dataset document (``docs/predictors/sources/``)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable source id (snake case).")
    title: str
    provider: str
    product: str
    licence: str
    licence_url: Optional[str] = None
    access_url: Optional[str] = None
    citation: Optional[str] = None
    version: Optional[str] = None
    notes: Optional[str] = None
    references: list[dict[str, str]] = Field(default_factory=list)
    markdown: Optional[str] = Field(
        default=None,
        description="Markdown filename relative to the sources/ directory.",
    )

    @field_validator("id")
    @classmethod
    def _id_is_snake(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or " " in cleaned:
            raise ValueError(f"source id must be a non-empty snake token, got {value!r}")
        return cleaned


class TemporalInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    notes: Optional[str] = None


class PipelineInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rebuild_step: str
    code: Optional[str] = None


class LayerDoc(BaseModel):
    """Machine-readable predictor layer document (YAML sidecar)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Band / variable id matching GeoTIFF descriptions.")
    title: str
    kind: LayerKind
    family: str
    summary: str
    shows: list[str] = Field(default_factory=list)
    does_not_show: list[str] = Field(default_factory=list)
    units: Optional[str] = None
    native_resolution_m: Optional[float] = None
    model_resolution_m: float = 100.0
    crs: str = "EPSG:27700"
    temporal: Optional[TemporalInfo] = None
    source_id: Optional[str] = Field(
        default=None,
        description="Id of a docs/predictors/sources/*.yml entry for licence/URL metadata.",
    )
    pipeline: Optional[PipelineInfo] = None
    derived_from: list[str] = Field(
        default_factory=list,
        description="Parent layer ids for lineage; do not copy parent caveats.",
    )
    processing: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    references: list[dict[str, str]] = Field(default_factory=list)
    markdown: Optional[str] = Field(
        default=None,
        description="Markdown filename relative to the layers/ directory.",
    )

    @field_validator("id")
    @classmethod
    def _id_is_snake(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or " " in cleaned:
            raise ValueError(f"layer id must be a non-empty snake token, got {value!r}")
        return cleaned

    @model_validator(mode="after")
    def _lineage_matches_kind(self) -> "LayerDoc":
        if self.kind == LayerKind.DERIVED and not self.derived_from:
            raise ValueError("derived layers must list derived_from parents")
        if self.kind == LayerKind.SOURCE and self.derived_from:
            raise ValueError("source layers must not list derived_from")
        return self


class PredictorCatalog(BaseModel):
    """Index of documented predictor layer and source ids."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    description: str = ""
    layers: list[str]
    sources: list[str] = Field(default_factory=list)

    @field_validator("layers")
    @classmethod
    def _unique_layer_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("catalog layer ids must be unique")
        return value

    @field_validator("sources")
    @classmethod
    def _unique_source_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("catalog source ids must be unique")
        return value
