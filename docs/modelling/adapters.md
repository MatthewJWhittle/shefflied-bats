# Model adapters

`sdm train`, `sdm predict`, and `sdm tune` resolve modelling methods through a small **adapter registry** instead of CLI `if trainer == …` branches.

## Contract

| Piece | Role |
|-------|------|
| `ModelAdapter` | Protocol: `train` → `TrainBundle`, `load_predictor`, optional Optuna `suggest_hyperparameters` / `eval_trial` |
| `TrainBundle` | Per-output `TrainingResults` (+ reserved `joint_artifact` for multi-output methods) |
| `Predictor` | Scoring surface used after load (`predict_proba` today for MaxEnt) |
| `ADAPTERS` | Registry keyed by `model.type` / package `trainer` |

Implementation: [`sdm/models/adapters.py`](../../sdm/models/adapters.py).

## MaxEnt (default)

- Config: omit `model.type` or set `model.type: maxent` in `model_config.yml`.
- Train writes `trainer: maxent` into each `package.json` and `run_metadata.json`.
- Packages remain elapid / sklearn `Pipeline` pickles (`model.pkl`) — same layout as before.
- Predict treats a missing `trainer` field as **maxent** so older runs still load.

## Adding a method

1. Implement a class with the `ModelAdapter` surfaces (tune may leave `supports_tune = False` until ready).
2. Register it: `ADAPTERS["your_type"] = YourAdapter()`.
3. Document package / joint artifact layout in [model-package-contract.md](../model-package-contract.md).

No new `if trainer ==` ladders in the train/predict/tune commands.

## DeepMaxent (#111)

[#111](https://github.com/MatthewJWhittle/shefflied-bats/pull/111) should land as a **`DeepMaxentAdapter`** on this seam: joint train + joint predict from one artifact. Do **not** ship façade / fake-Pipeline pickles as the long-term predict path. See the project cleanup plan for sequencing after this foundation merges.
