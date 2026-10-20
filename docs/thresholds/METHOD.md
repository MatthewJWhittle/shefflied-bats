# How the suitability cut-offs are set (20 Oct 2026)

1. Each model is trained several times, each time leaving out one group of records (a "fold"). The left-out records are scored by a model that never saw them.
2. In each fold, the cut-off is set at the score of the lowest 5% of the left-out presence records, so about 95% of unseen records fall above it.
3. The fold cut-offs are combined by taking the median, which is less thrown by one odd fold than the lowest or highest.
4. Fold models and the final model don't score on quite the same scale (the final model rates its own training records higher). So rather than copying the score, we copy the share of records caught: we work out what share of held-out records the median cut-off caught, then set the final model's cut-off at the score that catches the same share of its own training records.
5. That final score is the published cut-off. Anything above it is mapped as suitable.

Flags: models with AUC (area under the ROC curve) below 0.75 are marked model-limited, and models with fewer than 50 records are marked as having few records. Their cut-offs should not be trusted until the model itself is fixed.

Files: each training run writes ``thresholds.csv`` / ``thresholds.json`` in the run folder next to the model packages (cut-off, share caught, area mapped, flags), plus a per-package ``thresholds.json``. Code: ``sdm/models/core/threshold.py`` (``p5_median_transfer``). Models: retrain-5 on the rebuild-5 layers.

Fallback: an overfit final model scores almost all its own training records near the top, so step 4 pushes its cut-off to about 1.0. When the transferred cut-off is 0.95 or more, use the median fold cut-off from step 3 instead. For now that applies to Brandt's roost and whiskered in flight, until those models are fixed.
