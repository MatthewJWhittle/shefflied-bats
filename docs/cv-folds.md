# Cross-validation fold methods

Training scores each model with geographic cross-validation. The fold method is
selectable so methods can be compared on accuracy, overfitting (train versus
held-out gap), cut-offs and maps. The default is **`blocks`**: 10 km squares
dealt into 5 folds. `kmeans` and `buffered_loo` stay selectable.

## Methods

| Method | What it does | Typical use |
|--------|----------------|-------------|
| `blocks` | Squares of `block_size_m` (default 10 km) are assigned to `block_n_folds` (default 5) folds. Among many seeded random assignments, the run keeps the one that best balances presence counts while giving every fold both presence and background. Background inherits its block's fold. | **Default** spatial block CV |
| `kmeans` | Clusters point coordinates with KMeans into `n_folds` areas (elapid `GeographicKFold`). | Baseline / comparison runs |
| `buffered_loo` | Hold out each presence in turn. Drop training points within `buffer_m` (default 10 km) of it. Score that presence against background points inside the same buffer (also held out). Other nearby presences are excluded from train and test for that fold. | Thin species |

Seeds: fold construction uses `model.cv.seed` when set, otherwise `model.seed`. The same seed must give the same folds twice.

Skipped folds are never silent: the log warns with the fold number and the reason (for example only one class in train or test).

### Buffered LOO presence warning

`buffered_loo` is meant for thin species. When a model uses that method and its
presence-record count **exceeds** `buffered_loo_warn_above` (default **25**),
training logs a **warning per model** with the count, the threshold, and that
`blocks` or `kmeans` is the recommended method at that sample size. This is
guidance only: training still runs unchanged.

## Config (`model_config.yml`)

```yaml
model:
  seed: 42
  cv:
    method: blocks          # blocks | kmeans | buffered_loo
    n_folds: 3              # kmeans
    # seed: 42              # optional; defaults to model.seed
    block_size_m: 10000     # blocks
    block_n_folds: 5        # blocks
    block_n_tries: 500      # blocks
    buffer_m: 10000         # buffered_loo
    buffered_loo_warn_above: 25  # warn (only) when presence count exceeds this
```

## CLI

```bash
# Default (10 km blocks, 5 folds)
sdm train

# Explicit spatial blocks
sdm train --cv-method blocks
sdm train --cv-method blocks --cv-block-size-m 10000 --cv-block-n-folds 5

# Buffered leave-one-presence-out
sdm train --cv-method buffered_loo
sdm train --cv-method buffered_loo --cv-buffer-m 10000
sdm train --cv-method buffered_loo --cv-buffered-loo-warn-above 25

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

Cut-offs follow [thresholds/METHOD.md](thresholds/METHOD.md): pool every
held-out presence score across folds, then take the **5th percentile** of that
pool (`pooled_oof_percentile`).

For **`buffered_loo`**, each fold holds out one presence; those held-out scores
are pooled and the same 5th-percentile rule applies.
