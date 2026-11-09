# 0003: Train package write lives on the trainer adapter

Date: 9 Nov 2026. Status: Accepted.

## Modelling

## Context
`sdm train` resolves a `ModelAdapter` for fit, but MaxEnt package shape
(`model.pkl`, `package.json`, package-local sidecars) was still authored in the
train command (`save_models`). Shared `TrainingData` also required
`maxent_config`, forcing a MaxEnt-shaped field on every trainer.

## Options
1. Leave package write in the train command; keep the adapter as fit-only.
2. Move package write (and package-local sidecars) onto the trainer adapter;
   command keeps run orchestration only. Chosen.
3. Move the whole train command behind the adapter. Rejected: run folder layout,
   indexes, and run-level thresholds are shared orchestration, not trainer-specific.

## Decision
- The adapter (or MaxEnt code only it calls) writes each **model package** into
  directories the train command creates.
- The train command owns **run** orchestration: `runs/<id>/`, indexes,
  run-level thresholds, and `run_metadata.json`.
- Package-local sidecars (for example validation scores and fold assignments)
  move with package write; run-level files stay in the command.
- Shared `TrainingData` does not carry `maxent_config` or a generic
  `trainer_config`. MaxEnt hyperparameters are adapter-local input.
- `package.json` stays schema version 1; MaxEnt packages may still record
  `maxent_config` on disk.

## Why
So the train seam earns its keep under the deletion test, MaxEnt package shape
stays local to MaxEnt, and a later trainer (for example DeepMaxent) does not
inherit a hollow registry plus a MaxEnt-shaped shared type.

## Revisit when
- A trainer needs a different on-disk package contract (schema bump).
- Run-level vs package-local artefact split changes.
