"""Distance to coast is excluded from model feature selection (#75)."""

from __future__ import annotations

import csv
from pathlib import Path

from sdm.models.maxent.maxent_model import ActivityType, get_feature_config
from sdm.utils.io import load_variables_config
from sdm.utils.sdm_config import resolve_sdm_config_root

COAST = "bgs_coast_distance_to_coast"


def _variables_config_paths() -> list[Path]:
    roots = [
        Path(resolve_sdm_config_root("variables_config.yml")),
        *Path("data/sdm_config").glob("*/variables_config.yml"),
        *Path("data/sdm_tuning").glob("*/variables_config.yml"),
    ]
    return sorted({p.resolve() for p in roots if p.is_file()})


def test_no_variables_config_lists_distance_to_coast() -> None:
    paths = _variables_config_paths()
    assert paths, "expected committed variables_config.yml files"
    for path in paths:
        variables = load_variables_config(str(path)).variables
        assert COAST not in variables, f"{path} still lists {COAST}"


def test_default_activity_feature_configs_omit_coast() -> None:
    for activity in ActivityType:
        variables = get_feature_config(activity).variables
        assert COAST not in variables, f"{activity} default still lists {COAST}"


def test_ev_clusters_selected_excludes_coast_from_model_stack() -> None:
    csv_path = Path("data/ev_clusters_selected.csv")
    rows = list(csv.DictReader(csv_path.open(newline="")))
    coast_rows = [r for r in rows if r["column_name"] == COAST]
    assert len(coast_rows) == 1, "coast band should remain listed for provenance"
    assert coast_rows[0]["include"] == "0"
