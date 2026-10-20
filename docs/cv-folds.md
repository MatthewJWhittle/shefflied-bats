# Cross-validation fold methods

Training scores each model with geographic cross-validation. The fold method is
selectable so methods can be compared on accuracy, overfitting (train versus
held-out gap), cut-offs and maps. The default stays **`kmeans`** until a method
is chosen for production.

## Methods

| Method | What it does | Typical use |
|--------|----------------|-------------|
| `kmeans` | Clusters point coordinates with KMeans into `n_folds` areas (elapid `GeographicKFold`). Same behaviour as before. | Baseline / current runs |
| `blocks` | Squares of `block_size_m` (default 10 km) are assigned to `block_n_folds` (default 5) folds. Among many seeded random assignments, the run keeps the one that best balances presence counts while giving every fold both presence and background. Background inherits its block's fold. | Spatial block CV |
| `buffered_loo` | Hold out each presence in turn. Drop training points within `buffer_m` (default 10 km) of it. Score that presence against background points inside the same buffer (also held out). Other nearby presences are excluded from train and test for that fold. | Thin species |

Seeds: fold construction uses `model.cv.seed` when set, otherwise `model.seed`. The same seed must give the same folds twice.

Skipped folds are never silent: the log warns with the fold number and the reason (for example only one class in train or test).

## Config (`model_config.yml`)

```yaml
model:
  seed: 42
  cv:
    method: kmeans          # kmeans | blocks | buffered_loo
    n_folds: 3              # kmeans
    # seed: 42              # optional; defaults to model.seed
    block_size_m: 10000     # blocks
    block_n_folds: 5        # blocks
    block_n_tries: 500      # blocks
    buffer_m: 10000         # buffered_loo
```

## CLI

```bash
# Baseline (default)
sdm train

# Spatial blocks
sdm train --cv-method blocks
sdm train --cv-method blocks --cv-block-size-m 10000 --cv-block-n-folds 5

# Buffered leave-one-presence-out
sdm train --cv-method buffered_loo
sdm train --cv-method buffered_loo --cv-buffer-m 10000

# KMeans with an explicit fold count
sdm train --cv-method kmeans --cv-n-folds 3
```

CLI flags override the matching keys under `model.cv`.

## Run metadata and package outputs

- `run_metadata.json` records `cv` (method and parameters).
- Each model `package.json` records `fold`, `metrics.fold_method`, `metrics.fold_presence_counts`, and how many folds were scored.
- `validation_scores.parquet` keeps per-point held-out scores with a `fold` column.
- `fold_assignments.parquet` stores a fold label for every training point used in CV.

## Suitability cut-offs

Cut-offs still follow [thresholds/METHOD.md](thresholds/METHOD.md): per-fold 5th-percentile of held-out presence scores, then the median across folds.

For **`buffered_loo`**, each fold holds out one presence, so that fold's 5th percentile is that presence's held-out score. The published cut-off is therefore the **median of the held-out presence scores**.
