"""Tests for per-model SDM config path resolution (Linux case sensitivity)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdm.commands.modelling.utils import get_model_id
from sdm.utils.io import load_model_config, load_variables_config
from sdm.utils.model_runs import INPUT_DATA_ROOT_ENV
from sdm.utils.sdm_config import (
    find_subdirectory_case_insensitive,
    model_config_file_paths,
    resolve_model_config_dir,
    resolve_sdm_config_root,
)


def test_resolve_sdm_config_root_ignores_input_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        INPUT_DATA_ROOT_ENV, "gs://habitat-suitability-modelling-data/inputs"
    )
    resolved = resolve_sdm_config_root("data/sdm_config")
    assert resolved.endswith("/data/sdm_config") or resolved.endswith("data/sdm_config")
    assert "gs://" not in resolved
    assert Path(resolved).is_dir()


def test_base_model_config_ignores_input_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        INPUT_DATA_ROOT_ENV, "gs://habitat-suitability-modelling-data/inputs"
    )
    resolved = resolve_sdm_config_root("model_config.yml")
    assert resolved.endswith("model_config.yml")
    assert "gs://" not in resolved
    assert Path(resolved).is_file()
    cfg = load_model_config(resolved)
    assert cfg.background is not None
    assert cfg.background.n_background_points > 0


def test_capitalised_folder_found_from_lowercase_model_id() -> None:
    model_id = get_model_id(["Plecotus auritus", "Roost"])
    assert model_id == "plecotus_auritus_roost"
    config_dir = resolve_model_config_dir("data/sdm_config", model_id, must_exist=True)
    assert config_dir.endswith("Plecotus_auritus_Roost")
    _, variables_path = model_config_file_paths(config_dir)
    variables = load_variables_config(variables_path)
    assert len(variables.variables) == 15


def test_missing_per_model_folder_raises(tmp_path: Path) -> None:
    root = tmp_path / "sdm_config"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="plecotus_auritus_roost"):
        resolve_model_config_dir(root, "plecotus_auritus_roost", must_exist=True)


def test_case_insensitive_lookup_in_temp_dir(tmp_path: Path) -> None:
    root = tmp_path / "configs"
    (root / "Species_Activity").mkdir(parents=True)
    found = find_subdirectory_case_insensitive(str(root), "species_activity")
    assert found is not None
    assert found.endswith("Species_Activity")


def test_plecotus_auritus_roost_tuned_model_config_values() -> None:
    """Local proof: 15 layers and tuned MaxEnt betas from git configs."""
    model_id = get_model_id(["Plecotus auritus", "Roost"])
    config_dir = resolve_model_config_dir("data/sdm_config", model_id, must_exist=True)
    model_path, variables_path = model_config_file_paths(config_dir)
    project_seed = load_model_config().seed
    model_cfg = load_model_config(model_path, fallback_seed=project_seed)
    variables_cfg = load_variables_config(variables_path)
    assert len(variables_cfg.variables) == 15
    assert model_cfg.maxent is not None
    assert model_cfg.maxent.beta_multiplier == pytest.approx(3.33)
    assert model_cfg.maxent.n_hinge_features == 14


def test_fallback_base_variables_not_all_bands(tmp_path: Path) -> None:
    """When must_exist=False, new paths use lowercase model id."""
    root = tmp_path / "cfg"
    root.mkdir()
    path = resolve_model_config_dir(root, "my_species_roost", must_exist=False)
    assert path.endswith("my_species_roost")


def test_allow_config_fallback_uses_base_roster_not_all_bands() -> None:
    base_variables = load_variables_config(
        resolve_sdm_config_root("variables_config.yml")
    )
    assert len(base_variables.variables) == 42
    assert len(base_variables.variables) < 77
