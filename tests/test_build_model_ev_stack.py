"""Post-merge model EV stack (notebook feature-engineering reproduction)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
import yaml

from sdm.commands.data_preparation.processing.build_model_ev_stack import (
    apply_layer_clipping,
    build_model_ev_stack,
    canonicalize_band_name,
    default_build_settings,
    load_selected_variable_names,
    rename_ev_bands,
)


def test_canonicalize_band_names() -> None:
    assert canonicalize_band_name("landcover_arable") == "ceh_landcover_arable"
    assert canonicalize_band_name("coastal_distance_to_coast") == (
        "bgs_coast_distance_to_coast"
    )
    assert canonicalize_band_name("bgs_coast") == "bgs_coast_distance_to_coast"
    assert canonicalize_band_name("climate_bio_bio_4") == "climate_bioclim_bio_4"
    assert canonicalize_band_name("climate_bioclim_4") == "climate_bioclim_bio_4"
    assert canonicalize_band_name("vom_height_mean_100m") == "vom_vegetation_height_mean"
    assert (
        canonicalize_band_name("vom_vegetation_height_vom_height_max_100m")
        == "vom_vegetation_height_max"
    )
    assert (
        canonicalize_band_name("vom_vegetation_height_mean_500m")
        == "vom_vegetation_height_mean_500m"
    )


def test_apply_layer_clipping_fixed_and_quantile() -> None:
    y = np.arange(4.0)
    x = np.arange(4.0)
    ds = xr.Dataset(
        {
            "bgs_coast_distance_to_coast": (("y", "x"), np.full((4, 4), 9000.0)),
            "terrain_stats_curvature": (
                ("y", "x"),
                np.linspace(-10, 10, 16).reshape(4, 4),
            ),
        },
        coords={"y": y, "x": x},
    )
    original_min = float(ds["terrain_stats_curvature"].min())
    original_max = float(ds["terrain_stats_curvature"].max())
    clipped = apply_layer_clipping(ds)
    assert float(clipped["bgs_coast_distance_to_coast"].max()) == 7000.0
    assert float(clipped["terrain_stats_curvature"].min()) > original_min
    assert float(clipped["terrain_stats_curvature"].max()) < original_max


def _write_synthetic_all_evs(path: Path, band_names: list[str]) -> None:
    height = width = 40
    data = {
        name: (("y", "x"), np.random.default_rng(0).random((height, width)).astype(np.float32))
        for name in band_names
    }
    # Give distance layers values above the clip max so clipping is observable.
    for name in (
        "bgs_coast_distance_to_coast",
        "os_distance_distance_to_buildings",
        "os_distance_distance_to_major_roads",
    ):
        if name in data:
            data[name] = (("y", "x"), np.full((height, width), 8500.0, dtype=np.float32))
    ds = xr.Dataset(
        data,
        coords={
            "y": np.arange(height, dtype=np.float64) * 100.0 + 400_000.0,
            "x": np.arange(width, dtype=np.float64) * 100.0 + 400_000.0,
        },
    )
    ds = ds.rio.write_crs("EPSG:27700")
    for var in ds.data_vars:
        ds[var].attrs["long_name"] = var
        ds[var].rio.write_nodata(np.nan, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.rio.to_raster(path)


def _all_sdm_config_variables() -> set[str]:
    root = Path("data/sdm_config")
    needed: set[str] = set()
    for path in root.glob("*/variables_config.yml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        needed.update(payload.get("variables") or [])
    return needed


def test_synthetic_stack_covers_all_sdm_config_variables(tmp_path: Path) -> None:
    selected_csv = Path("data/ev_clusters_selected.csv")
    selected = load_selected_variable_names(selected_csv)
    # Include curvature (clipped then dropped by include=0) plus one alias form.
    base_bands = list(dict.fromkeys(selected + ["terrain_stats_curvature"]))

    all_evs = tmp_path / "all-evs.tif"
    out = tmp_path / "evs-to-model.tif"
    _write_synthetic_all_evs(all_evs, base_bands)

    settings = default_build_settings()
    settings["selected_variables_csv"] = str(selected_csv)

    _, used = build_model_ev_stack(all_evs, out, build_settings=settings)
    assert used["layer_clipping"]["enabled"] is True
    assert used["layer_clipping"]["layers"]["bgs_coast_distance_to_coast"]["max"] == 7000.0

    result = rxr.open_rasterio(out, masked=True, band_as_variable=True)
    # Band names may be band_1..; prefer long_name attrs from write.
    names = []
    for var in result.data_vars:
        long_name = result[var].attrs.get("long_name", var)
        names.append(canonicalize_band_name(str(long_name)))
    # Also accept dataset variable names if already canonical.
    names.extend(canonicalize_band_name(str(v)) for v in result.data_vars)
    band_set = set(names)

    needed = _all_sdm_config_variables()
    missing = sorted(needed - band_set)
    assert not missing, f"Rebuilt stack missing config variables: {missing}"

    # Clipping applied on coast distance before subset/multiscale.
    # Re-open via rename helper for a numeric check on the written stack.
    renamed = rename_ev_bands(result)
    if "bgs_coast_distance_to_coast" in renamed.data_vars:
        assert float(renamed["bgs_coast_distance_to_coast"].max()) == 7000.0


def test_build_settings_recorded_shape() -> None:
    settings = default_build_settings()
    assert settings["layer_clipping"]["enabled"] is True
    for layer in (
        "bgs_coast_distance_to_coast",
        "os_distance_distance_to_buildings",
        "os_distance_distance_to_major_roads",
        "terrain_stats_curvature",
    ):
        assert layer in settings["layer_clipping"]["layers"]


def test_selected_csv_include_count() -> None:
    keep = load_selected_variable_names("data/ev_clusters_selected.csv")
    with open("data/ev_clusters_selected.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(keep) == sum(1 for row in rows if row["include"].strip() == "1")
