"""Fold-stratified subsampling (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from sdm.occurrence.subsample import stratified_subsample


def _frame(n_pres=200, n_bg=2000, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    cls = np.array([1] * n_pres + [0] * n_bg)
    folds = rng.integers(0, n_folds, size=len(cls))
    return pd.DataFrame({"class": cls}), folds


def test_fraction_is_applied_per_fold_and_class():
    df, folds = _frame()
    keep = stratified_subsample(
        df, folds, presence_fraction=0.5, background_fraction=0.25, random_state=1
    )
    for f in range(5):
        for cls, frac in [(1, 0.5), (0, 0.25)]:
            n = ((folds == f) & (df["class"] == cls)).sum()
            k = ((folds == f) & (df["class"] == cls) & keep).sum()
            assert k == round(n * frac)


def test_minimum_keeps_rare_classes_whole():
    df, folds = _frame(n_pres=30)
    keep = stratified_subsample(
        df, folds, presence_fraction=0.1, background_fraction=0.1,
        min_presence=50, min_background=500, random_state=1,
    )
    assert keep[df["class"].to_numpy() == 1].all()
    assert 480 <= keep[df["class"].to_numpy() == 0].sum() <= 520


def test_same_seed_same_rows_and_validation():
    df, folds = _frame()
    a = stratified_subsample(df, folds, background_fraction=0.3, random_state=7)
    b = stratified_subsample(df, folds, background_fraction=0.3, random_state=7)
    np.testing.assert_array_equal(a, b)
    assert stratified_subsample(df, folds, random_state=7).all()
    with pytest.raises(ValueError):
        stratified_subsample(df, folds, background_fraction=0, random_state=7)
