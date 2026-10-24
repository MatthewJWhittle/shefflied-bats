# Shared MLflow run tracking

Models can train on more than one computer. This design records each run in one
shared place (settings, held-out scores, cut-offs, plots, and model files) so
either computer — or an assistant on request — can compare runs.

Closes the product outcome in [#93](https://github.com/MatthewJWhittle/shefflied-bats/issues/93).
Builds on the draft helpers from [#28](https://github.com/MatthewJWhittle/shefflied-bats/pull/28).

## Architecture

| Piece | Choice | Why |
| --- | --- | --- |
| Tracking server | **None** | Each computer talks straight to the database. |
| Run metadata | Neon Postgres (free tier) | Shared, durable, SQLAlchemy dialect MLflow already supports. |
| Plots / model files | Private GCS prefix under the existing data bucket | Keeps artefacts off the `inputs/` and `runs/` layer trees; MLflow paths use unique experiment/run IDs so nothing is overwritten. |
| Database password | Google Secret Manager (local machines); `MLFLOW_DB_PASSWORD` for cloud agents | Never in git. Fetched when a run starts via Application Default Credentials, or injected by Cursor secret settings. |
| Database login | Dedicated role (`mlflow_tracking`) | Can only reach MLflow tables. |
| Public git | Code, `mlflow.settings.example.yml` (empty values), gitignore for `mlflow.settings.yml` | Safe for an open-source repo. |

Local defaults in `config.yml` keep tracking **off** until a real settings file exists.
With tracking off, the password is never fetched and Secret Manager is not contacted.
CI runs [gitleaks](https://github.com/gitleaks/gitleaks) on every pull request so a
mistakenly committed settings file cannot land silently.

## Local setup (after cloud resources exist)

1. Copy `mlflow.settings.example.yml` → `mlflow.settings.yml` (gitignored).
2. Fill host, database name, tracking user, Secret Manager ids, and `artifact_root`.
3. Set `mlflow.enabled: true` in `config.yml` (or `enabled: true` in the settings file).
4. Ensure Google ADC can read the secret (or set `MLFLOW_DB_PASSWORD` for agents).
5. Run `sdm train` / `sdm tune` as usual.

Environment overrides (tests and one-off local SQLite):

| Variable | Purpose |
| --- | --- |
| `MLFLOW_ENABLED` | `false` skips tracking |
| `MLFLOW_TRACKING_URI` | e.g. `sqlite:////tmp/mlflow.db` for offline tests |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name |
| `MLFLOW_ARTIFACT_ROOT` | Local directory or `gs://…` prefix |
| `MLFLOW_DB_PASSWORD` | Password for cloud agents (Cursor secrets); never commit |

Compare runs with the MLflow UI pointed at the same backend (once Neon is live):

```bash
uv run mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

## GCS layout (future path only — not written by this PR)

Documented destination for artefacts (no live bucket writes from this change):

`gs://habitat-suitability-modelling-data/mlflow-artifacts/<experiment_id>/<run_id>/…`

Separate from `inputs/`, `runs/`, and `published/`. See [data-layout.md](data-layout.md).

## Matt approval — exact create steps

**Do not create these resources until Matt has approved this list.**

### 1. Neon Postgres (tracking backend)

1. In the Neon console, create a project (region close to the GCP project, e.g. Europe).
2. Create a database named `mlflow` (or agreed name).
3. Create a role named `mlflow_tracking` with a strong password. Do **not** use the project owner role for day-to-day tracking.
4. Grant that role only what MLflow needs on that database (connect + create/use tables in the public schema for MLflow’s SQLAlchemy store). Prefer a role that cannot read other Neon databases or project settings.
5. Note the host, port, database name, and user for `mlflow.settings.yml`. Keep the password out of git.

### 2. Google Secret Manager

1. Enable the Secret Manager API on the GCP project if it is not already enabled.
2. Create a secret, suggested id: `mlflow-tracking-db-password`.
3. Store the `mlflow_tracking` role password as the secret payload (one version).
4. Grant `roles/secretmanager.secretAccessor` on that secret to:
   - the owner Google account used on local machines, and
   - the `model-run` service account (see `infra/iam.tf`),
   so both can fetch the password via ADC when a run starts.
5. Record `project_id`, `secret_id`, and `version` (`latest`) in the local settings file only.

### 3. GCS private prefix

1. On bucket `habitat-suitability-modelling-data`, use prefix `mlflow-artifacts/` (create by writing the first object later; no need for a special folder resource).
2. Confirm public access prevention remains enforced (already on the bucket).
3. Ensure `model-run` retains object create/read under this prefix (existing `roles/storage.objectAdmin` on the bucket covers it). Do **not** grant broader bucket-admin rights.
4. Treat the prefix as append-only for humans: MLflow run IDs make paths unique; do not point `artifact_root` at `inputs/` or `runs/`.

### 4. Cloud agents (Cursor)

1. After Neon + Secret Manager exist, add a Cursor secret named to match `MLFLOW_DB_PASSWORD` with the same password value.
2. Agents must never receive the password via the repository or committed files.

### 5. Turn tracking on

1. Fill `mlflow.settings.yml` from the example (still gitignored).
2. Set `mlflow.enabled: true` in `config.yml` (or rely on `enabled: true` in settings).
3. Smoke-test: one short train on each computer; confirm both runs appear when searching the shared experiment.

## Safety

- No secrets, licensed data, or site-specific coordinates in this document or in git.
- This PR does **not** create Neon, Secret Manager, or GCS objects, and does **not** write to `gs://`.
