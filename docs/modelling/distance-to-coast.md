# Distance to coast — dropped from model features

## Decision

`bgs_coast_distance_to_coast` is **not** used as a model feature. It is excluded from:

- every `data/sdm_config/*/variables_config.yml` and `data/sdm_tuning/*/variables_config.yml` that listed it (including Nathusius' pipistrelle roost)
- the fallback roster `variables_config.yml`
- activity-type defaults in `sdm/models/maxent/maxent_model.py`
- the model EV selection CSV `data/ev_clusters_selected.csv` (`include = 0`)

The coastal-distance prep step may still write `coastal_distance.tif` into a rebuild folder for QA or other pipelines. That does not put the band into `evs-to-model.tif` while `include` stays 0.

## Why

From the environmental-variables data-quality checks and the ecology review on [#70](https://github.com/MatthewJWhittle/shefflied-bats/pull/70) (parent programme [#71](https://github.com/MatthewJWhittle/shefflied-bats/issues/71); this change is [#75](https://github.com/MatthewJWhittle/shefflied-bats/issues/75)):

- The layer is clipped at 7,000 m. About **85%** of study-area cells sit at that cap, so the band is nearly constant inland and carries little information across most of the area.
- The mechanisms the layer was meant to stand for (wind exposure, low/open vegetation) are already represented by other variables kept in the stack (notably wind and vegetation height).
- The ecology review therefore recommends dropping it, including from the older Nathusius' pipistrelle config where it had been kept as a proxy.

## Role

**Modeller** — feature selection only; no ecology claims beyond the review cited above.
