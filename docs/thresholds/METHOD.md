# How the suitability cut-offs are set (20 Oct 2026)

1. Each model is trained several times, each time leaving out one group of records (a "fold"). The left-out records are scored by a model that never saw them. Fold construction is selectable (`kmeans`, `blocks`, `buffered_loo`); see [cv-folds.md](../cv-folds.md).
2. In each fold, the cut-off is set at the score of the lowest 5% of the left-out presence records, so about 95% of unseen records fall above it.
3. The fold cut-offs are combined by taking the median, which is less thrown by one odd fold than the lowest or highest.
4. That median score is the published cut-off. It is applied directly to the final model's suitability scores: anything above it is mapped as suitable.

**Buffered leave-one-out.** With `buffered_loo`, each fold holds out a single presence (scored against held-out background inside the buffer). The 5th percentile of one score is that score, so the median across folds is the median of the held-out presence scores. The same `p5_fold_median` code path applies.

Flags: models with AUC (area under the ROC curve) below 0.75 are marked model-limited, and models with fewer than 50 records are marked as having few records. Their cut-offs should not be trusted until the model itself is fixed.

Files: each training run writes ``thresholds.csv`` / ``thresholds.json`` in the run folder next to the model packages (cut-off, share caught, area mapped, flags), plus a per-package ``thresholds.json``. Code: ``sdm/models/core/threshold.py`` (``p5_fold_median``).
