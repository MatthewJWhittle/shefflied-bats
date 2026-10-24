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

## Keeping and cleaning up runs

Native MLflow only — no scheduler, no extra clean-up script, and **no GCS
bucket lifecycle rule** on `mlflow-artifacts/`. A lifecycle rule only sees
object age, not whether a model is registered, so it could delete artefacts
that still back a Model Registry version.

What goes into MLflow (small files only): the fitted model, the results table,
settings / feature lists, metrics, and optional small plots. Big prediction
maps stay in the existing `runs/` (and local predictions) tree — training and
predict paths do **not** upload prediction GeoTIFFs or input layers as MLflow
artefacts.

### Measure one run’s artefact size (read-only)

Find a run id in the MLflow UI (open the run; the id is in the header / URL), or
via the CLI against the same tracking URI:

```bash
uv run mlflow runs list -e <experiment_name>
# or: uv run python -c "import mlflow; mlflow.set_tracking_uri('…'); print(mlflow.search_runs())"
```

The experiment id is on the experiment page (or in `mlflow experiments list`).
Then, from a machine with Google ADC (read-only list/du — no writes):

```bash
gcloud storage du -s \
  "gs://habitat-suitability-modelling-data/mlflow-artifacts/<experiment_id>/<run_id>/"
```

If the UI shows a different Artifact URI for that run, use that path instead.
Expect a small total (model + table + settings), not prediction-map scale.

### 1. Keep a good model (Model Registry)

Promote a keeper by registering it in the MLflow Model Registry — in the UI, or
with `mlflow.register_model` — optionally with an alias such as `champion`.
Registered versions are the ones you must not delete in the steps below.

### 2. Soft-delete unwanted runs (UI)

Periodically delete old unwanted runs in the MLflow UI. Skip any run that still
backs a registered model version. Deleting in the UI only moves the run to the
`deleted` lifecycle stage; artefacts remain in GCS until the next step.

### 3. Permanently remove deleted runs (`mlflow gc`)

Run this from **Matt’s Mac** (trusted machine with Google ADC that can write/
delete under `gs://habitat-suitability-modelling-data/mlflow-artifacts/`).
Verified against the project lockfile: **MLflow 3.6.0**.

`mlflow gc` needs:

| Requirement | Why |
| --- | --- |
| `--backend-store-uri` | Neon Postgres URI (same SQLAlchemy store the clients use). |
| `--tracking-uri` or `MLFLOW_TRACKING_URI` | Required by MLflow 3.6.0 even with a direct DB backend; use the same URI. |
| Google ADC for the bucket | Artefacts live at each run’s `gs://…` URI; there is no tracking-server proxy, so **`--artifacts-destination` is not used**. |

Build the URI in the shell from local settings + a password from the environment
or Secret Manager. **Never write the password to disk or git.**

```bash
# Password: env / Cursor secret, or fetch once via ADC (do not echo or redirect to a file).
# export MLFLOW_DB_PASSWORD=…   # or:
# MLFLOW_DB_PASSWORD="$(gcloud secrets versions access latest \
#   --secret=mlflow-tracking-db-password --project=habitat-suitability-modelling)"

# Fill MLFLOW_DB_HOST / MLFLOW_DB_NAME / MLFLOW_DB_USER from gitignored
# mlflow.settings.yml (values are not written in this doc).
ENCODED_PW="$(python3 -c 'import os,urllib.parse as u; print(u.quote_plus(os.environ["MLFLOW_DB_PASSWORD"]))')"
BACKEND="postgresql+psycopg://${MLFLOW_DB_USER}:${ENCODED_PW}@${MLFLOW_DB_HOST}:5432/${MLFLOW_DB_NAME}?sslmode=require"

uv run mlflow gc --backend-store-uri "$BACKEND" --tracking-uri "$BACKEND"

unset MLFLOW_DB_PASSWORD ENCODED_PW BACKEND
```

That permanently removes soft-deleted runs’ metadata from Neon and their
artefacts under `gs://habitat-suitability-modelling-data/mlflow-artifacts/`.
**Do not** soft-delete (and therefore do not `gc`) a run that still backs a
registered model version — that would remove the model’s files.

Optional flags (same CLI): `--older-than` (e.g. `30d`), `--run-ids`,
`--experiment-ids`. See `uv run mlflow gc --help`.

### Project budget alert (separate)

A **£5/month** GCP project budget alert is recommended so storage growth is
noticed early. That is a **manual cloud console step for Matt** to approve and
create — not part of this PR, and **not** added as Terraform here.

## Safety

- No secrets, licensed data, or site-specific coordinates in this document or in git.
- This PR does **not** create Neon, Secret Manager, or GCS objects, and does **not** write to `gs://`.
