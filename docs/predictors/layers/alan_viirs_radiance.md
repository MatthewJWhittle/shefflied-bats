# Artificial light at night (`alan_viirs_radiance`)

Annual VIIRS night-lights radiance on the model grid.

## What it shows

Relative artificial brightness (nW/cm²/sr) from the EOG VNL **2025**
`average_masked` composite.

## What it does not show

Lamp type, under-canopy light, or lighting in the year of each bat record.

## Source and licence

EOG VNL v2.2 (CC-BY 4.0; attribute Elvidge et al.; mark changes). Files live in
the project bucket / local cache. Licence and access: [eog_viirs_vnl](../sources/eog_viirs_vnl.md).

Full operational detail: [alan-layer.md](../alan-layer.md).

## Pipeline

`alan` rebuild step (auto-skip if source missing) → merge → selected
(`include=1`). Species feature lists are a separate enablement step.

## Role

**Data** — ALAN predictor; long-form ops remain in `docs/alan-layer.md`.
