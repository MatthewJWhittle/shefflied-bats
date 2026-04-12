"""Tests for resolving packaged vs flat model pickle paths."""

import pickle
from pathlib import Path

import pytest

from sdm.utils.io import load_pickled_model, resolve_trained_model_path


def test_resolve_accepts_package_model_pkl(tmp_path):
    pkg = tmp_path / "species_activity_slug"
    pkg.mkdir()
    pkl = pkg / "model.pkl"
    pkl.write_bytes(pickle.dumps({"kind": "dummy"}))
    assert resolve_trained_model_path(pkl) == pkl.resolve()


def test_resolve_accepts_package_directory(tmp_path):
    pkg = tmp_path / "species_activity_slug"
    pkg.mkdir()
    pkl = pkg / "model.pkl"
    pkl.write_bytes(pickle.dumps({"kind": "dummy"}))
    assert resolve_trained_model_path(pkg) == pkl.resolve()


def test_load_pickled_model_via_directory(tmp_path):
    pkg = tmp_path / "my_species_in_flight"
    pkg.mkdir()
    obj = {"x": 1}
    (pkg / "model.pkl").write_bytes(pickle.dumps(obj))
    assert load_pickled_model(pkg) == obj


def test_resolve_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_trained_model_path(tmp_path / "nope.pkl")
