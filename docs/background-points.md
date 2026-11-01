# Background (comparison) points

Live `sdm train` draws a shared pool of background points, then prepares them
per model. Every step is a config knob. Defaults match today's behaviour so an
unchanged config stays byte-identical to previous main.

Product context and the #103 trial: [issue #103](https://github.com/MatthewJWhittle/shefflied-bats/issues/103).

## Knobs

| Key | Default | Notes |
|-----|---------|--------|
| `n_background_points` | `8000` | Shared pool size from the surface |
| `surface` | `all_bat` | `all_bat` (bias-weighted density) or `uniform` |
| `background_method` | `contrast` | Floor method on the density surface |
| `background_value` | `0.025` | Contrast fraction of peak density |
| `sigma` | `6.5` | Gaussian smooth on the density surface |
| `transform_method` | `presence` | Count transform before smoothing |
| `surface_resolution_m` | spatial resolution (`100`) | Density-grid cell size |
| `presence_exclusion_m` | `500` | Drop background within this of presence; `0` = off |
| `presence_exclusion_max_m` | `null` (∞) | Optional max distance ring |
| `background_thin_m` | `2000` | One background point per cell; `0` = off |
| `presence_thin_m` | `2000` | One presence point per cell; `0` = off |
| `cap_ratio` | `10` | Max background = ratio × thinned presence; `null` = off |
| `min_background` / `max_background` | `null` | Optional absolute clamps after the ratio |
| `distance_weights` | `true` | Distance weights on presence |
| `background_distance_weights` | `true` | Distance weights on background |
| `sample_weight_n_neighbors` | `10` | Neighbours for distance weights (live default; was CLI-only) |
| `class_balance` | `true` | MaxEnt `class_weights: balanced` when true, else `1.0` |
| `seed` | `model.seed` | Sampling / thinning / cap RNG |

CV folds stay under `model.cv` (default 10 km blocks, 5 folds).

## Where to set them

1. **Global defaults** — `config.yml` → `background:`
2. **Model / run overrides** — `model_config.yml` → `model.background`
3. **CLI** (optional) — `--d-min`, `--d-max`, `--sample-weight-n-neighbors`, `--n-background-points`

Later layers win. Per-species folders under `data/sdm_config/` may also set
`model.background` when loaded as the model config path.

Legacy `model.sampling.grid_size_m` still fills presence/background thin when
those keys are unset. Legacy `sampling.background.factor` fills `cap_ratio`
when unset. Absolute `sampling.background.min_bg` / `max_bg` are **not** applied
on the live path (they were previously ignored); use
`background.min_background` / `max_background` instead.

## #103 trial packaging

Ready-made file: [`model_config.background-as-drawn.yml`](../model_config.background-as-drawn.yml)

- 10 000 points from the all-bat contrast surface
- Exclusion, background thinning, 10× cap, and background distance weights **off**
- Presence thinning, class balance, and 10 km block CV **kept**

```bash
# 10k trial
sdm train --model-config-path model_config.background-as-drawn.yml

# 50k stability: change n_background_points to 50000 in that file, then:
sdm train --model-config-path model_config.background-as-drawn.yml
```

Do not train or publish from CI; Matt runs on the box against the 23 Oct layers.

## Run metadata

`run_metadata.json` records every resolved background knob under `background`
(and `cv` / `seed` as before). Published prediction `run.json` picks up a
different config hash when these change.
