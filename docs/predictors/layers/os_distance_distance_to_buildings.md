# Distance to buildings (`os_distance_distance_to_buildings`)

Metres to the nearest OS VMD building feature on the model grid.

## What it shows / does not show

Proximity to mapped buildings — not height, type, or roost quality.

## Source and licence

OS Vector Map District (OGL). Keep raw tiles out of git as with other OS caches.

## Pipeline

`os` step → `os-distance-to-feature.tif` → merge → clip 0–7000 m in
`build_model_ev_stack`.

## Caveats

Distance cap and mapping lag; see YAML.

## Role

**Data** — OS distance predictor.
