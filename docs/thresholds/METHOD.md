# How the suitability cut-offs are set (20 Oct 2026)

## Modelling rationale

### One rule for every model

Every model uses the same cut-off rule: pool all held-out presence scores, then take the **10th percentile**. Per-model or few-record special cases are hard to defend and to document, so they are not used. Models that are too weak for any cut-off are flagged separately (below) rather than given a different percentile.

### Why pool held-out scores

Spatial-block folds leave thin models with about 6–9 records per fold. A per-fold 5th percentile then sits on 0–1 records and is pushed close to zero (in a blocked-fold trial, Brandt's roost came out at a cut-off of about 0.11 under the old median-of-fold-p5 rule). Pooling every out-of-fold presence score uses the full held-out sample at once, so the cut-off is not dominated by the lowest fold.

### Fit only on held-out predictions

The cut-off is fit only on held-out (out-of-fold) presence scores. Scores from the model that trained on a point are optimistic for that point; fitting a cut-off on them would leak evaluation information into the threshold.

### Why the 10th percentile

The 10th percentile is the conventional presence-only threshold that allows about 10% training omission — the familiar “10th percentile training presence” option in MaxEnt and the spirit of Pearson et al.’s fixed T10 threshold for presence-only Maxent maps (Pearson et al. 2007; Phillips & Dudík 2008). It allows for about 1 in 10 records sitting in poor habitat or being errors (bats passing through, locational error). The 5th percentile keeps more area but trusts the records more. Matt chose the 10th.

That choice **shrinks** well-recorded suitable maps compared with today’s published cut-offs. In a blocked-fold trial under the pooled 10th percentile, common pipistrelle roost went from about 34% to about 20% of the map, and soprano pipistrelle roost from about 50% to about 23%. That shrinkage is the accepted trade-off, not a side effect.

Trial suitable-map shares under the pooled 10th percentile (blocked folds; to be confirmed by a real-data rerun after merge):

| Model | Suitable map share (trial) |
|-------|----------------------------|
| Brandt's roost | about 16% |
| Common pipistrelle roost | about 20% |
| Soprano pipistrelle roost | about 23% |
| Leisler's roost | about 17% |

### Data-limited models (no cut-off)

A per-model `model_limited: true` flag in that model's `model_config.yml` means the model publishes **no cut-off**: no threshold value and no thresholded map. The reason is recorded in the threshold metadata (`no_cutoff_reason: model_limited`). Whiskered in flight (`Myotis_mystacinus_In_flight` / `myotis_mystacinus_in_flight`) is flagged this way — limited by its data. Brandt's roost keeps a cut-off and goes to the per-model ecology checks (#62) rather than getting a special rule. Soft diagnostic flags (for example AUC below 0.75, or fewer than 50 records) may still appear on models that do publish a cut-off; they do not by themselves suppress the cut-off or change the percentile.

## Procedure

1. Each model is trained several times, each time leaving out one group of records (a "fold"). The left-out records are scored by a model that never saw them. Fold construction defaults to **10 km spatial blocks** dealt into **5 folds** (`blocks`); `kmeans` and `buffered_loo` remain selectable — see [cv-folds.md](../cv-folds.md). Spatial blocking for evaluation follows the same idea as geographically structured / block cross-validation in the Maxent and ecological modelling literature (Radosavljevic & Anderson 2014; Roberts et al. 2017).
2. Every held-out presence score from every fold is pooled into one list. The cut-off is the **10th percentile of that pool** — not the median of per-fold percentiles. Only held-out predictions are used; training points never inform the cut-off.
3. That pooled-percentile score is the published cut-off. It is applied directly to the final model's suitability scores: anything above it is mapped as suitable.

**Buffered leave-one-out.** With `buffered_loo`, each fold holds out a single presence (scored against held-out background inside the buffer). Pooling those held-out presence scores and taking the 10th percentile is the same code path (`pooled_oof_percentile`). Leave-one-out remains available as a cross-check for thin models (see the `buffered_loo_warn_above` guidance in [cv-folds.md](../cv-folds.md)); it is not the main method.

Files: each training run writes ``thresholds.csv`` / ``thresholds.json`` in the run folder next to the model packages (cut-off or null, share caught, area mapped, flags, percentile used), plus a per-package ``thresholds.json``. Publishing refuses to overwrite existing threshold files. Code: ``sdm/models/core/threshold.py`` (``pooled_oof_percentile``).

## References

- Liu, C., Berry, P.M., Dawson, T.P. & Pearson, R.G. (2005) Selecting thresholds of occurrence in the prediction of species distributions. *Ecography*, **28**, 385–393. https://doi.org/10.1111/j.0906-7590.2005.03957.x
- Liu, C., White, M. & Newell, G. (2013) Selecting thresholds for the prediction of species occurrence with presence-only data. *Journal of Biogeography*, **40**, 778–789. https://doi.org/10.1111/jbi.12058
- Pearson, R.G., Raxworthy, C.J., Nakamura, M. & Peterson, A.T. (2007) Predicting species distributions from small numbers of occurrence records: a test case using cryptic geckos in Madagascar. *Journal of Biogeography*, **34**, 102–117. https://doi.org/10.1111/j.1365-2699.2006.01594.x
- Phillips, S.J. & Dudík, M. (2008) Modeling of species distributions with Maxent: new extensions and a comprehensive evaluation. *Ecography*, **31**, 161–175. https://doi.org/10.1111/j.0906-7590.2008.5203.x
- Radosavljevic, A. & Anderson, R.P. (2014) Making better Maxent models of species distributions: complexity, overfitting and evaluation. *Journal of Biogeography*, **41**, 629–643. https://doi.org/10.1111/jbi.12227
- Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schröder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F. & Dormann, C.F. (2017) Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, **40**, 913–929. https://doi.org/10.1111/ecog.02881
