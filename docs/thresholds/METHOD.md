# How the suitability cut-offs are set (20 Oct 2026)

1. Each model is trained several times, each time leaving out one group of records (a "fold"). The left-out records are scored by a model that never saw them. Fold construction defaults to **10 km spatial blocks** dealt into **5 folds** (`blocks`); `kmeans` and `buffered_loo` remain selectable — see [cv-folds.md](../cv-folds.md).
2. Every held-out presence score from every fold is pooled into one list. The cut-off is a **low percentile of that pool** — not the median of per-fold percentiles. Only held-out predictions are used; training points never inform the cut-off.
3. Models with **fewer than 50** presence records use the **10th** percentile of the pool; other models use the **5th**. Both the record-count line and the two percentiles are configurable under `model.threshold` in `model_config.yml` (defaults: `few_records_max: 50`, `few_records_percentile: 10`, `percentile: 5`).
4. That pooled-percentile score is the published cut-off. It is applied directly to the final model's suitability scores: anything above it is mapped as suitable.

**Buffered leave-one-out.** With `buffered_loo`, each fold holds out a single presence (scored against held-out background inside the buffer). Pooling those held-out presence scores and taking the configured percentile is the same code path (`pooled_oof_percentile`).

**Model-limited (no cut-off).** A per-model `model_limited: true` flag in that model's `model_config.yml` means the model publishes **no cut-off**: no threshold value and no thresholded map. The reason is recorded in the threshold metadata (`no_cutoff_reason: model_limited`). Soft diagnostic flags (for example AUC below 0.75, or few records) may still appear on models that do publish a cut-off; they do not by themselves suppress the cut-off.

Files: each training run writes ``thresholds.csv`` / ``thresholds.json`` in the run folder next to the model packages (cut-off or null, share caught, area mapped, flags, percentile used), plus a per-package ``thresholds.json``. Code: ``sdm/models/core/threshold.py`` (``pooled_oof_percentile``).
