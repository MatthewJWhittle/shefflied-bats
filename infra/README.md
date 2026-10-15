# Infrastructure

Terraform for a small Google Cloud setup that stores model inputs and run outputs.
Data isn't stored in this repo.

## What it creates

| Resource | Purpose |
| --- | --- |
| Data bucket | `inputs/`, `runs/`, `published/` prefixes. Uniform access, public access prevention, versioning (old versions kept 30 days). |
| State bucket | Terraform remote state. Versioned, not public. |
| Budget | Monthly budget alert at 50% and 100%. |
| `model-run` service account | Reads `inputs/` and `runs/`, creates new objects under `runs/`. Cannot delete or overwrite. Can list file names in `published/` but can't read or change anything there. |
| `terraform-deploy` service account | Used by GitHub Actions to apply changes, via Workload Identity Federation (no keys). |
| Workload identity pool/provider | Trusts only this repository, `main`, in the `production` environment. |

## Bootstrap (owner, once, locally — before merging)

The project ID and region are not committed. They live as GitHub Actions
variables on the `production` environment (`GCP_PROJECT_ID`, `GCP_REGION`); the
workflow passes them as `TF_VAR_project_id` / `TF_VAR_region` and fails if unset.
Locally, put them in your gitignored `terraform.tfvars`.

Nothing project-specific is in code. All of these are `production` environment
variables:

| Variable | Set when | Used for |
| --- | --- | --- |
| `GCP_PROJECT_ID` | now | `TF_VAR_project_id` |
| `GCP_REGION` | now | `TF_VAR_region` |
| `TF_STATE_BUCKET` | after bootstrap | `terraform init -backend-config="bucket=..."` (partial backend config) |
| `GCP_WIF_PROVIDER` | after bootstrap (`terraform output`) | Workload Identity provider for GitHub sign-in |
| `GCP_DEPLOY_SA` | after bootstrap (`terraform output`) | deploy service account email |
| `GCP_BUCKET_NAME`, `GCP_BILLING_ACCOUNT` | after bootstrap | `TF_VAR_bucket_name`, `TF_VAR_billing_account` |
| `MODEL_RUN_IMPERSONATOR_EMAIL` | now | `TF_VAR_model_run_impersonator_email` (token creator on `model-run`) |

The apply job fails with a clear error if any required variable is unset.

Do this before the first merge to `main`. The apply job needs the state
bucket and the workload identity provider, which only the bootstrap creates.

1. Link billing to the project. Sign in: `gcloud auth application-default login`.
2. Enable the APIs Terraform needs (replace `<project-id>`):

   ```
   gcloud services enable --project=<project-id> \
     cloudresourcemanager.googleapis.com serviceusage.googleapis.com \
     iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
     storage.googleapis.com billingbudgets.googleapis.com \
     cloudbilling.googleapis.com
   ```

3. `cp terraform.tfvars.example terraform.tfvars` and fill in (gitignored).
4. Comment out the `backend "gcs"` block in `versions.tf`, then
   `terraform init && terraform apply`.
5. Restore the backend block and migrate state:
   `terraform init -migrate-state -backend-config="bucket=<state-bucket-name>"`.
6. Grant the deploy identity `roles/billing.costsManager` on the billing account.
7. In GitHub, set `production` environment variables from `terraform output`
   (`GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`) plus
   `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_BUCKET_NAME`, `TF_STATE_BUCKET`,
   `GCP_BILLING_ACCOUNT`, `MODEL_RUN_IMPERSONATOR_EMAIL`.

After that, changes go through pull requests and are applied on merge.

## Deploy flow

- Pull requests touching `infra/`: format and validate only (no cloud access).
- Push to `main`: the `apply` job waits for approval in the `production`
  environment, then runs `terraform plan` and `apply` with short-lived credentials.

## Local runs with the run identity

Default: no key files. Run locally with your own sign-in and
service account impersonation:

```
gcloud auth application-default login --impersonate-service-account=<run-sa-email>
```

Optional exception (off by default, owner decision): a key for the run identity
on one trusted machine. It is not created by Terraform, so it never enters state.
If approved, create it with `gcloud iam service-accounts keys create`, keep it
outside the repo, and delete it when no longer needed.

## References

- Google provider: https://registry.terraform.io/providers/hashicorp/google/latest/docs
- Bucket resource: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket
- GCS IAM conditions: https://cloud.google.com/storage/docs/access-control/iam#conditions
- GCS IAM roles: https://cloud.google.com/storage/docs/access-control/iam-roles
- Public access prevention: https://cloud.google.com/storage/docs/public-access-prevention
- Workload Identity Federation with deployment pipelines: https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines
- GitHub OIDC claims: https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- Terraform GCS backend: https://developer.hashicorp.com/terraform/language/backend/gcs
- Budget resource: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/billing_budget
