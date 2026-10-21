"""Overlapping / proxy environmental variables are thinned (#72)."""

from __future__ import annotations

import csv
from pathlib import Path

from sdm.commands.data_preparation.processing.build_model_ev_stack import (
    DEFAULT_MULTISCALE_WINDOWS_CELLS,
    default_build_settings,
)
from sdm.models.maxent.maxent_model import ActivityType, get_feature_config
from sdm.utils.io import load_variables_config
from sdm.utils.sdm_config import resolve_sdm_config_root

DROPPED = frozenset(
    {
        "climate_stats_temp_ann_var",
        "climate_stats_temp_mat_avg",
        "climate_stats_prec_ann_var",
        "climate_bioclim_bio_2",
        "climate_bioclim_bio_3",
        "climate_bioclim_bio_7",
        "climate_bioclim_bio_8",
        "climate_bioclim_bio_9",
        "os_distance_distance_to_minor_roads",
        "os_cover_buildings",
        "vom_vegetation_height_mean",
    }
)

CLIMATE_KEPT = frozenset(
    {
        "climate_stats_temp_ann_avg",
        "climate_bioclim_bio_4",
        "climate_stats_prec_ann_avg",
        "climate_stats_wind_ann_avg",
    }
)

BUILDING_KEPT = "os_distance_distance_to_buildings"


def _variables_config_paths() -> list[Path]:
    roots = [
        Path(resolve_sdm_config_root("variables_config.yml")),
        *Path("data/sdm_config").glob("*/variables_config.yml"),
        *Path("data/sdm_tuning").glob("*/variables_config.yml"),
    ]
    return sorted({p.resolve() for p in roots if p.is_file()})


def test_no_variables_config_lists_dropped_overlap_bands() -> None:
    paths = _variables_config_paths()
    assert paths, "expected committed variables_config.yml files"
    for path in paths:
        variables = load_variables_config(str(path)).variables
        for name in variables:
            assert name not in DROPPED, f"{path} still lists {name}"
            assert not name.endswith("_1000m"), f"{path} still lists 1000 m copy {name}"
            assert not name.endswith("_1500m"), f"{path} still lists 1500 m copy {name}"
            assert not name.startswith("vom_vegetation_height_mean"), (
                f"{path} still lists VOM mean {name}"
            )


def test_default_activity_feature_configs_omit_dropped_bands() -> None:
    for activity in ActivityType:
        variables = get_feature_config(activity).variables
        for name in DROPPED:
            assert name not in variables, f"{activity} default still lists {name}"
        for name in variables:
            assert not name.endswith("_1000m")
            assert not name.endswith("_1500m")


def test_ev_clusters_selected_marks_dropped_bands_exclude() -> None:
    csv_path = Path("data/ev_clusters_selected.csv")
    rows = list(csv.DictReader(csv_path.open(newline="")))
    by_name = {r["column_name"]: r["include"].strip() for r in rows}
    for name in DROPPED:
        assert name in by_name, f"{name} should remain listed for provenance"
        assert by_name[name] == "0", f"{name} should have include=0"
    for name in CLIMATE_KEPT:
        assert by_name[name] == "1", f"{name} should remain include=1"
    assert by_name[BUILDING_KEPT] == "1"
    assert by_name["os_cover_buildings"] == "0"
    assert by_name["os_distance_distance_to_minor_roads"] == "0"


def test_multiscale_defaults_use_500m_window_only() -> None:
    assert DEFAULT_MULTISCALE_WINDOWS_CELLS == (10,)
    settings = default_build_settings()
    assert settings["multiscale"]["windows_cells"] == [10]
