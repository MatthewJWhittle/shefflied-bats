# Habitat suitability modelling

Domain language for this repository’s species distribution / habitat suitability
work. Architecture words (module, seam, adapter, …) live in design practice, not
here.

## Language

**Model package**:
On-disk portable artefacts for one species × activity model (`model.pkl`,
`package.json`, and optional sidecars).
_Avoid_: model folder (ambiguous with run), pickle (too narrow)

**Run**:
One training (or tune) output directory under the models root, holding many
model packages plus run-level indexes and metadata.
_Avoid_: experiment (MLflow sense), job

**Trainer**:
The modelling method used for a run or model package (for example maxent or
deepmaxent), recorded on the package and run metadata.
_Avoid_: backend, algorithm (vague)

**Suitability score**:
Continuous habitat suitability in [0, 1] produced by scoring a model on
environmental predictors.
_Avoid_: probability (unless literally class probability), prediction (broader)

**Rebuild step**:
One ordered unit of regenerating environmental layers that `--skip` / `--only`
can name (including layer pieces, merge, and stack).
_Avoid_: job, task, pipeline stage
