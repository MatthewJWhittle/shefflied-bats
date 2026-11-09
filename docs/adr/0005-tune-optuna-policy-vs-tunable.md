# 0005: Tune keeps Optuna policy; Tunable owns suggest and eval

Date: 9 Nov 2026. Status: Accepted.

## Modelling

## Context
`sdm tune` resolved a Tunable adapter and called `require_tune_support`, then
the Optuna objective still used command-local MaxEnt `suggest_maxent_hyperparameters`
and `eval_model`. Adapter Tunable methods were unused on the live path.
Fold labels and per-fold prune lived only on the command eval helper.

## Options
1. Leave suggest/eval duplicated in the tune command; adapter remains a gate only.
2. Route live trials through Tunable suggest/eval (including fold labels and
   prune), keep Optuna study / feature pick / penalties / data prep in the
   command. Chosen.
3. Move feature pick, penalties, and the whole objective into the adapter.
   Rejected: those are shared tuning policy, not MaxEnt-specific; stuffing them
   into MaxEnt would bloat Tunable or duplicate policy per trainer.

## Decision
- Live trials call the adapter’s `suggest_hyperparameters` and `eval_trial`.
- MaxEnt suggest/eval has one locality on the adapter side; command duplicates
  go away once wired.
- Tunable eval supports the live fold-label and per-fold prune behaviour.
- The tune command keeps Optuna study setup, feature picking, composite
  penalties, data prep, and result writers.
- MaxEnt hyperparameters remain adapter-local (see ADR 0003); not a field on
  shared `TrainingData`.

## Why
So Tunable survives the deletion test, MaxEnt hyperparameter knowledge is not
duplicated, and the next trainer can reuse the same Optuna / feature-pick /
penalty loop without inheriting MaxEnt CV guts—or swallowing shared tuning
policy.

## Revisit when
- A trainer needs a radically different trial objective (not AUC-minus-penalties).
- Feature picking becomes method-specific rather than shared policy.
