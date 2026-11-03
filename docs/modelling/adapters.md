# Model adapters

`sdm train`, `sdm predict`, and `sdm tune` resolve modelling methods through a small **adapter registry** instead of CLI `if trainer == …` branches.

## Contract

| Piece | Role |
|-------|------|
| `ModelAdapter` | Protocol: `train` → `TrainBundle`, `load_predictor` |
| `Tunable` | Optional Optuna surface: `suggest_hyperparameters` / `eval_trial` (implement when tune is ready) |
| `TrainBundle` | Per-output `TrainingResults` + the `training_data` used to fit |
| `Predictor` | Scoring surface used after load (`predict_proba` today for MaxEnt) |
| `ADAPTERS` | Registry keyed by `model.type` / package `trainer` |

Implementation: [`sdm/models/adapters.py`](../../sdm/models/adapters.py). MaxEnt fit helpers live under [`sdm/models/maxent/training.py`](../../sdm/models/maxent/training.py) so adapters do not import the Typer command module.

## MaxEnt (default)

- Config: omit `model.type` or set `model.type: maxent` in `model_config.yml`.
- Train writes `trainer: maxent` into each `package.json` and `run_metadata.json`.
- Packages remain elapid / sklearn `Pipeline` pickles (`model.pkl`) — same layout as before.
- Predict treats a missing `trainer` field as **maxent** so older runs still load.

## Adding a method

1. Implement `ModelAdapter` (`train`, `load_predictor`). Add `Tunable` methods only when Optuna tune is ready.
2. Register it: `ADAPTERS["your_type"] = YourAdapter()`.
3. Document package / joint artifact layout in [model-package-contract.md](../model-package-contract.md).

No new `if trainer ==` ladders in the train/predict/tune commands. Call sites thread the already-resolved adapter rather than re-selecting `"maxent"` by string.

## DeepMaxent (#111)

[#111](https://github.com/MatthewJWhittle/shefflied-bats/pull/111) should land as a **`DeepMaxentAdapter`** on this seam: joint train + joint predict from one artifact. Extra run metadata / joint artifacts belong on that adapter’s `TrainBundle` when they exist — not as empty placeholders on MaxEnt. Do **not** ship façade / fake-Pipeline pickles as the long-term predict path.
