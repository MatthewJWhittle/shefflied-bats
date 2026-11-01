# DeepMaxent (experimental)

Joint multi-output residual MLP with DeepMaxent / MaxEnt loss
([Ryckewaert et al. 2024](https://arxiv.org/abs/2412.19217)). Optional alternative
to per-model elapid MaxEnt. **MaxEnt stays the default** (`model.type: maxent`).

## Install

```bash
uv sync --extra dl
```

Base installs and default CI do not pull PyTorch. CI job `deepmaxent (dl extra)`
runs synthetic CPU tests only.

## Enable

In `model_config.yml`:

```yaml
model:
  type: deepmaxent
  deepmaxent:
    device: auto   # MPS → CUDA → CPU; or mps / cuda / cpu
    hidden_size: 250   # e.g. 512
    n_hidden_layers: 2 # e.g. 4 (residual skips between hidden layers)
    dropout: 0.0       # hidden activations; 0 = off
    input_dropout: 0.0 # input feature channels in train mode only; 0 = off
    learning_rate: 0.0002
    batch_size: 250
    weight_decay: 0.0003
    epochs: 100
    n_seeds: 10
    use_species_weights: false
    balanced_batches: false          # guarantee k presences per output / batch
    presences_per_output: 4          # only used when balanced_batches is true
```



Then the usual train / predict / publish commands:

```bash
# Mac (Apple MPS when available)
uv sync --extra dl
uv run sdm train

# CPU-only
# model.deepmaxent.device: cpu
uv run sdm train

uv run sdm predict --run <run-id>
# publish when ready (not part of this experimental landing):
# uv run sdm predict --run <run-id> --publish
```

`run_metadata.json` records `trainer`, resolved `device`, and DeepMaxent knobs.
Each model package still has `model.pkl` + `package.json` so `sdm predict --run`
is unchanged. Packages require the `dl` extra to unpickle.

Training uses **PyTorch Lightning** (`Trainer` + `LightningModule`). Per-epoch
history (train loss; for each CV fold, held-out val loss and per-output AUC) comes
from Lightning’s **CSVLogger** (aggregated to `deepmaxent_training_history.csv` in
the run folder) and **MLFlowLogger** when MLflow tracking is on. Device selection
is Lightning `accelerator="auto"` (CPU here; MPS on Apple Silicon when available).

## Design sketch

- One shared MLP (2×250 ReLU by default, residual skips, BatchNorm over sites,
  optional hidden `dropout` and optional `input_dropout` on feature channels in
  train mode only) with **one logit per model** (roost and in-flight as separate
  outputs). Larger nets (e.g. 512 units × 4 layers) are config-only.
- Loss: batched DeepMaxent (paper eq. 4) — per-species softmax over sites in the
  mini-batch, occurrence-normalised weights, epsilon when a species is absent.
- Background: same all-bat sampler as today’s MaxEnt train path (reuse #103
  settings when that lands; no hard dependency).
- CV: shared 10 km spatial blocks for every output so held-out scores line up
  with MaxEnt. Cut-off: pooled held-out 10th percentile.
- Features: **union** of per-model feature lists (shared trunk). Departure from
  per-model MaxEnt subsets — required for one joint representation.
- `use_species_weights` (inverse-frequency) is **off by default** (rare→common
  pull risk); multiplies each species term in the count-weighted loss when on.
- `balanced_batches` (**off by default**): each train batch includes
  `presences_per_output` (default 4) presence rows for every output (with
  replacement if rare), remainder background. Keeps batch size and steps/epoch
  unchanged. Oversampled presence sites also enter other outputs' softmax
  denominators as non-presences — eq. 4 already handles that; no extra correction.
- `input_dropout` (**default 0 / off**): drops input feature channels during
  training only (`nn.Dropout`; inactive at predict / eval).
- After each DeepMaxent train, response curves (sklearn average PD) are written
  under `{run}/{model_id}/partial_dependence/` as `pd__{feature}.csv` + `.png`
  for every output model. Artifacts stay in the run folder — never commit them
  (licensed site data).

## Helper scripts (real data; not CI)

- `scripts/deepmaxent_compare_scores.py` — MaxEnt vs DeepMaxent AUC; flags five rarest
- `scripts/deepmaxent_partial_dependence.py` — re-run PD for one model (train already writes them)
- `scripts/deepmaxent_seed_spread.py` — seed-spread workflow notes
- `scripts/deepmaxent_plot_history.py` — loss / AUC vs epoch from history CSV

### Three-seed comparison (real data; Matt)

```bash
uv sync --extra dl
# model_config.yml: model.type: deepmaxent  and  deepmaxent.n_seeds: 3
uv run sdm train
uv run python scripts/deepmaxent_compare_scores.py \
  --maxent-run data/sdm_models/runs/<maxent-run-id> \
  --deepmaxent-run data/sdm_models/runs/<deepmaxent-run-id> \
  --out deepmaxent_vs_maxent_scores.csv
```
