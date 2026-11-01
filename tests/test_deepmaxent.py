"""Synthetic CPU tests for experimental DeepMaxent (optional ``dl`` extra)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from sdm.models.deepmaxent.config import DeepMaxentParams, resolve_deepmaxent_params
from sdm.models.deepmaxent.data import build_joint_presence_background_table
from sdm.models.deepmaxent.device import resolve_torch_device
from sdm.models.deepmaxent.estimator import wrap_output_pipeline
from sdm.models.deepmaxent.history import save_training_history
from sdm.models.deepmaxent.loss import deepmaxent_loss
from sdm.models.deepmaxent.manual_fit import (
    fit_one_seed_manual,
    predict_logits_from_state,
)
from sdm.models.deepmaxent.network import build_residual_mlp
from sdm.models.deepmaxent.partial_dependence import (
    PD_SUBDIR,
    write_run_partial_dependence,
)
from sdm.models.deepmaxent.sampling import (
    BalancedPresenceBatchSampler,
    build_train_dataloader,
)
from sdm.models.deepmaxent.training import (
    _fit_one_seed_lightning,
    fit_ensemble,
    train_deepmaxent_joint,
)
from sdm.commands.modelling.utils import get_model_id
from sdm.models.maxent.folds import FoldMethod, FoldParams
from sdm.models.maxent.maxent_model import (
    DefaultMaxentConfig,
    create_maxent_pipeline,
)
from sdm.models.core.threshold import compute_pooled_oof_percentile
from sdm.types import ModelConfig, TrainingData
from sdm.utils.io import load_model_config


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny_params(**overrides) -> DeepMaxentParams:
    base = dict(
        device="cpu",
        hidden_size=16,
        n_hidden_layers=2,
        dropout=0.0,
        learning_rate=2e-3,
        batch_size=32,
        weight_decay=3e-4,
        epochs=40,
        n_seeds=2,
        epsilon=1e-6,
        use_species_weights=False,
        transform="cloglog",
        early_stopping_patience=None,
        seed=0,
    )
    base.update(overrides)
    return DeepMaxentParams(**base)


def _synthetic_species_gdf(
    rng: np.random.Generator,
    *,
    n_presence: int,
    n_background: int,
    coef: np.ndarray,
    feature_names: list[str],
    x0: float = 0.0,
) -> gpd.GeoDataFrame:
    """Presence-background table with a linear logit habitat signal."""
    n = n_presence + n_background
    X = rng.normal(size=(n, len(feature_names)))
    scores = X @ coef
    order = np.argsort(scores)[::-1]
    X = X[order]
    y = np.zeros(n, dtype=int)
    y[:n_presence] = 1
    data = {name: X[:, i] for i, name in enumerate(feature_names)}
    data["class"] = y
    data["sample_weight"] = np.ones(n)
    data["geometry"] = [
        Point(x0 + i * 100.0, (i % 7) * 100.0) for i in range(n)
    ]
    return gpd.GeoDataFrame(data, crs="EPSG:27700")


@pytest.fixture
def two_species_training_data() -> list[TrainingData]:
    rng = np.random.default_rng(7)
    feats = ["f1", "f2", "f3"]
    coef_a = np.array([1.5, -0.8, 0.2])
    coef_b = np.array([1.2, -0.5, 0.1])
    gdf_a = _synthetic_species_gdf(
        rng, n_presence=40, n_background=120, coef=coef_a, feature_names=feats, x0=0
    )
    gdf_b = _synthetic_species_gdf(
        rng, n_presence=35, n_background=120, coef=coef_b, feature_names=feats, x0=50_000
    )
    cfg = DefaultMaxentConfig(feature_types=["linear"], beta_multiplier=1.0)
    return [
        TrainingData(
            latin_name="Demo one",
            activity_type="Roost",
            occurrence=gdf_a,
            maxent_config=cfg,
            model_features=feats,
            model_limited=False,
        ),
        TrainingData(
            latin_name="Demo two",
            activity_type="In flight",
            occurrence=gdf_b,
            maxent_config=cfg,
            model_features=feats,
            model_limited=False,
        ),
    ]


def test_resolve_device_cpu():
    assert resolve_torch_device("cpu") == "cpu"


def test_deepmaxent_loss_shapes_and_finite():
    logits = torch.randn(20, 3, dtype=torch.float32)
    targets = torch.zeros(20, 3)
    targets[0, 0] = 1
    targets[5, 1] = 1
    loss = deepmaxent_loss(logits, targets, epsilon=1e-6)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_species_weights_change_loss_and_config_default():
    """Inverse-frequency weights scale the count-weighted loss; default stays off."""
    cfg = load_model_config()
    assert cfg.deepmaxent is not None
    assert cfg.deepmaxent.use_species_weights is False
    params = resolve_deepmaxent_params(cfg)
    assert params.use_species_weights is False
    assert params.balanced_batches is False
    assert params.presences_per_output == 4
    assert cfg.deepmaxent.balanced_batches is False
    assert cfg.deepmaxent.presences_per_output == 4

    logits = torch.randn(10, 2, dtype=torch.float32)
    targets = torch.zeros(10, 2)
    targets[:5, 0] = 1
    targets[5:, 1] = 1
    base = deepmaxent_loss(logits, targets, epsilon=1e-6)
    weighted = deepmaxent_loss(
        logits,
        targets,
        epsilon=1e-6,
        species_weights=np.array([2.0, 0.5]),
    )
    assert torch.isfinite(weighted)
    assert not torch.allclose(base, weighted)


def _synthetic_multi_output_Y(
    *,
    n_rows: int,
    n_outputs: int,
    presence_counts: list[int],
) -> np.ndarray:
    """Build a Y matrix with known presence counts and shared background rows."""
    if len(presence_counts) != n_outputs:
        raise ValueError("presence_counts length must match n_outputs")
    Y = np.zeros((n_rows, n_outputs), dtype=np.float64)
    cursor = 0
    for j, count in enumerate(presence_counts):
        if cursor + count > n_rows:
            raise ValueError("presence rows exceed n_rows")
        Y[cursor : cursor + count, j] = 1.0
        cursor += count
    # Remaining rows stay background (all-zero).
    return Y


def test_balanced_batches_contain_k_presences_per_output():
    """Every balanced batch has at least k presence rows for each output."""
    n_outputs = 5
    k = 4
    batch_size = 40
    Y = _synthetic_multi_output_Y(
        n_rows=200,
        n_outputs=n_outputs,
        # One rare output (2 < k → with-replacement), others common.
        presence_counts=[2, 20, 15, 12, 10],
    )
    X = np.random.default_rng(1).normal(size=(Y.shape[0], 3)).astype(np.float64)
    loader = build_train_dataloader(
        X,
        Y,
        batch_size=batch_size,
        seed=0,
        balanced_batches=True,
        presences_per_output=k,
    )
    assert len(loader) == int(np.ceil(Y.shape[0] / batch_size))
    for xb, yb in loader:
        assert xb.shape[0] == batch_size
        assert yb.shape == (batch_size, n_outputs)
        counts = (yb > 0).sum(dim=0)
        assert torch.all(counts >= k), f"presence counts {counts.tolist()} < k={k}"


def test_balanced_sampler_length_matches_default_steps():
    """Balanced path keeps the same number of batches per epoch as shuffle."""
    Y = _synthetic_multi_output_Y(
        n_rows=115, n_outputs=3, presence_counts=[3, 8, 12]
    )
    sampler = BalancedPresenceBatchSampler(
        Y, batch_size=25, presences_per_output=2, seed=0
    )
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(
        torch.zeros(Y.shape[0], 1),
        torch.as_tensor(Y, dtype=torch.float32),
    )
    default = DataLoader(ds, batch_size=25, shuffle=True, drop_last=False)
    assert len(sampler) == len(default)


def test_default_off_dataloader_matches_historical_shuffle_and_loss():
    """balanced_batches=False keeps historical shuffle batches and loss values."""
    rng = np.random.default_rng(3)
    n, p, k_out = 80, 4, 3
    X = rng.normal(size=(n, p)).astype(np.float64)
    Y = np.zeros((n, k_out), dtype=np.float64)
    Y[:10, 0] = 1
    Y[10:25, 1] = 1
    Y[25:40, 2] = 1
    seed = 17
    batch_size = 16

    # Historical construction (pre–balanced_batches path).
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(Y, dtype=torch.float32),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    historical = DataLoader(
        ds, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False
    )
    current = build_train_dataloader(
        X,
        Y,
        batch_size=batch_size,
        seed=seed,
        balanced_batches=False,
    )
    assert len(historical) == len(current)

    net = build_residual_mlp(p, k_out, hidden_size=8, n_hidden_layers=2, dropout=0.0)
    net.eval()
    for (xh, yh), (xc, yc) in zip(historical, current):
        assert torch.equal(xh, xc)
        assert torch.equal(yh, yc)
        with torch.no_grad():
            logits = net(xh)
        loss_h = deepmaxent_loss(logits, yh)
        loss_c = deepmaxent_loss(logits, yc)
        assert torch.equal(loss_h, loss_c)


def test_dropout_default_off_and_forward_runs():
    model0 = build_residual_mlp(4, 2, hidden_size=8, n_hidden_layers=2, dropout=0.0)
    model_d = build_residual_mlp(4, 2, hidden_size=8, n_hidden_layers=2, dropout=0.3)
    x = torch.randn(5, 4)
    model0.train()
    model_d.train()
    y0 = model0(x)
    y_d = model_d(x)
    assert y0.shape == (5, 2)
    assert y_d.shape == (5, 2)
    assert model0.dropout.p == 0.0
    assert model_d.dropout.p == 0.3
    assert model0.input_dropout.p == 0.0


def test_input_dropout_default_off_active_when_set():
    """input_dropout is off by default; when set, trains only (eval is deterministic)."""
    cfg = load_model_config()
    assert cfg.deepmaxent is not None
    assert cfg.deepmaxent.input_dropout == 0.0
    params = resolve_deepmaxent_params(cfg)
    assert params.input_dropout == 0.0

    torch.manual_seed(0)
    x = torch.randn(32, 4)
    model_off = build_residual_mlp(
        4, 2, hidden_size=8, n_hidden_layers=2, dropout=0.0, input_dropout=0.0
    )
    model_on = build_residual_mlp(
        4, 2, hidden_size=8, n_hidden_layers=2, dropout=0.0, input_dropout=0.5
    )
    assert model_off.input_dropout.p == 0.0
    assert model_on.input_dropout.p == 0.5

    model_on.train()
    torch.manual_seed(1)
    y_a = model_on(x).detach()
    torch.manual_seed(2)
    y_b = model_on(x).detach()
    assert not torch.allclose(y_a, y_b), "train-mode input dropout should randomise"

    model_on.eval()
    torch.manual_seed(1)
    y_e1 = model_on(x).detach()
    torch.manual_seed(2)
    y_e2 = model_on(x).detach()
    assert torch.allclose(y_e1, y_e2), "eval-mode input dropout must be inactive"


def test_large_network_512x4_fits_end_to_end(two_species_training_data):
    """Config sizes like 512 units × 4 residual layers train and predict."""
    table = build_joint_presence_background_table(two_species_training_data)
    params = _tiny_params(
        epochs=3,
        n_seeds=1,
        hidden_size=512,
        n_hidden_layers=4,
        batch_size=64,
        learning_rate=1e-3,
    )
    ensemble, states, device, history = fit_ensemble(
        table, params, n_seeds=1, fit_id="large"
    )
    assert device == "cpu"
    assert len(states) == 1
    # Residual layers present in state dict.
    hidden_keys = [k for k in states[0] if k.startswith("hidden.")]
    assert any("hidden.3." in k for k in hidden_keys)
    proba = ensemble.predict_proba_multi(table.features)
    assert proba.shape[1] == len(table.output_names)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert len(history) >= 1


def test_estimator_fits_and_predicts_unit_interval(two_species_training_data):
    table = build_joint_presence_background_table(two_species_training_data)
    params = _tiny_params(epochs=15, n_seeds=1)
    ensemble, _states, device, history = fit_ensemble(
        table, params, n_seeds=1, fit_id="unit"
    )
    assert device == "cpu"
    assert "train_loss" in history.columns
    assert history["train_loss"].notna().any()
    proba = ensemble.predict_proba_multi(table.features)
    assert proba.shape == (len(table.features), len(table.output_names))
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    pipe = wrap_output_pipeline(ensemble, table.output_names[0])
    out = pipe.predict_proba(table.features)
    assert out.shape == (len(table.features), 2)
    assert np.all(out[:, 1] >= 0.0) and np.all(out[:, 1] <= 1.0)


def test_lightning_matches_manual_loop_on_tiny_batch():
    """Lightning Trainer matches the hand-rolled Adam loop when one batch / epoch."""
    rng = np.random.default_rng(0)
    n, p, k = 24, 3, 2
    X = rng.normal(size=(n, p)).astype(np.float64)
    Y = np.zeros((n, k), dtype=np.float64)
    Y[:8, 0] = 1
    Y[8:16, 1] = 1
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    params = _tiny_params(
        epochs=12,
        n_seeds=1,
        batch_size=n,  # single batch → shuffle order irrelevant
        hidden_size=8,
        n_hidden_layers=2,
        learning_rate=5e-3,
        seed=11,
    )
    manual_state = fit_one_seed_manual(
        Xs, Y, params=params, device="cpu", seed=params.seed
    )
    lightning_state, history, _device = _fit_one_seed_lightning(
        Xs,
        Y,
        params=params,
        seed=params.seed,
        output_names=["a", "b"],
        fit_id="equiv",
    )
    assert len(history) >= 1
    logits_m = predict_logits_from_state(
        manual_state, Xs, n_outputs=k, params=params, device="cpu"
    )
    logits_l = predict_logits_from_state(
        lightning_state, Xs, n_outputs=k, params=params, device="cpu"
    )
    # Same loss + optimiser + single-batch epochs → close predictions.
    corr = float(np.corrcoef(logits_m.ravel(), logits_l.ravel())[0, 1])
    assert corr > 0.95, f"Lightning vs manual logit corr={corr:.3f}"
    assert np.allclose(logits_m, logits_l, atol=0.75, rtol=0.2)


def test_linear_synthetic_ranks_close_to_elapid_maxent(two_species_training_data):
    data = two_species_training_data[0]
    feats = data.model_features
    occ = data.occurrence
    X = occ[feats]
    y = occ["class"].to_numpy()
    w = occ["sample_weight"].to_numpy()

    maxent = create_maxent_pipeline(
        feature_names=feats,
        maxent_n_jobs=1,
        model_config=DefaultMaxentConfig(
            feature_types=["linear"],
            beta_multiplier=1.0,
            n_hinge_features=0,
            n_threshold_features=0,
        ),
    )
    maxent.fit(X, y, maxent__sample_weight=w)
    maxent_scores = maxent.predict_proba(X)[:, 1]

    table = build_joint_presence_background_table([data])
    params = _tiny_params(epochs=40, n_seeds=1, hidden_size=8, learning_rate=5e-3)
    ensemble, _, _, _ = fit_ensemble(table, params, n_seeds=1)
    pipe = wrap_output_pipeline(ensemble, table.output_names[0])
    dm_on_occ = pipe.predict_proba(X)[:, 1]
    corr, _p = spearmanr(maxent_scores, dm_on_occ)
    assert corr > 0.5, f"Spearman r={corr:.3f} too low vs elapid MaxEnt"


def test_joint_cv_threshold_history_and_predict(
    two_species_training_data, tmp_path: Path
):
    params = _tiny_params(epochs=8, n_seeds=1, batch_size=64)
    fold_params = FoldParams(
        method=FoldMethod.BLOCKS,
        n_folds=3,
        random_state=0,
        block_size_m=500.0,
        block_n_tries=50,
    )
    result = train_deepmaxent_joint(
        two_species_training_data,
        params,
        fold_params=fold_params,
        log_root=tmp_path / "lightning_logs",
    )
    assert len(result.results) == 2
    assert result.device == "cpu"
    assert result.training_history is not None
    assert len(result.training_history) > 0
    assert result.training_history["train_loss"].notna().any()
    # Fold fits should have validation metrics; final may not.
    fold_rows = result.training_history[
        result.training_history["fit_id"].astype(str).str.startswith("fold_")
    ]
    assert len(fold_rows) > 0
    assert fold_rows["val_loss"].notna().any()

    hist_path = save_training_history(result.training_history, tmp_path)
    assert hist_path.exists()
    loaded = pd.read_csv(hist_path)
    assert "epoch" in loaded.columns

    for tr in result.results:
        assert tr.success
        assert tr.final_model is not None
        assert tr.validation_scores is not None
        cut = compute_pooled_oof_percentile(tr.validation_scores, percentile=10.0)
        assert "threshold" in cut
        vs = tr.validation_scores
        if vs["class"].nunique() > 1:
            auc = roc_auc_score(vs["class"], vs["held_out_score"])
            assert 0.0 <= auc <= 1.0

    grid = result.joint_table.features.copy()
    for tr in result.results:
        pred = tr.final_model.predict_proba(grid)[:, 1]
        assert np.all((pred >= 0.0) & (pred <= 1.0))

    path = tmp_path / "run_metadata.json"
    path.write_text(
        json.dumps(
            {
                "trainer": "deepmaxent",
                "device": result.device,
                "deepmaxent": result.params.to_metadata(),
            },
            indent=2,
        )
    )
    assert json.loads(path.read_text())["device"] == "cpu"


def test_run_writes_partial_dependence_csv_and_png(
    two_species_training_data, tmp_path: Path
):
    """Every output model gets PD CSV+PNG under the run package folder."""
    params = _tiny_params(epochs=4, n_seeds=1, batch_size=64)
    fold_params = FoldParams(
        method=FoldMethod.BLOCKS,
        n_folds=2,
        random_state=0,
        block_size_m=500.0,
        block_n_tries=50,
    )
    result = train_deepmaxent_joint(
        two_species_training_data,
        params,
        fold_params=fold_params,
        log_root=tmp_path / "lightning_logs",
    )
    run_dir = tmp_path / "run"
    written = write_run_partial_dependence(
        run_dir,
        result.results,
        result.joint_table.features,
        feature_names=result.feature_names,
        grid_resolution=11,
    )
    assert len(written) == len(result.output_names)
    for tr in result.results:
        mid = get_model_id([tr.latin_name, tr.activity_type])
        pd_dir = run_dir / mid / PD_SUBDIR
        assert pd_dir.is_dir()
        for feat in result.feature_names:
            csv_path = pd_dir / f"pd__{feat}.csv"
            png_path = pd_dir / f"pd__{feat}.png"
            assert csv_path.is_file(), csv_path
            assert png_path.is_file(), png_path
            table = pd.read_csv(csv_path)
            assert feat in table.columns
            assert "suitability" in table.columns
            assert len(table) == 11
            assert png_path.stat().st_size > 0
