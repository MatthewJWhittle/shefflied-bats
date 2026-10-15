locals {
  objects_prefix = "projects/_/buckets/${var.bucket_name}/objects"
}

# ---------------------------------------------------------------------------
# Run identity: reads inputs/, writes new objects under runs/.
# Can list file names in published/ but can't read or change anything there. No delete, so existing objects cannot be overwritten.
# ---------------------------------------------------------------------------
resource "google_service_account" "run" {
  account_id   = "model-run"
  display_name = "Model run"
}

resource "google_storage_bucket_iam_member" "run_read_inputs" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.run.email}"

  # Listing is checked against the bucket, not a prefix, so object names
  # across the bucket are listable; object contents are readable only under inputs/.
  condition {
    title      = "inputs-read"
    expression = "resource.type == \"storage.googleapis.com/Bucket\" || resource.name.startsWith(\"${local.objects_prefix}/inputs/\")"
  }
}

resource "google_storage_bucket_iam_member" "run_create_runs" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.run.email}"

  condition {
    title      = "runs-create"
    expression = "resource.name.startsWith(\"${local.objects_prefix}/runs/\")"
  }
}

# ---------------------------------------------------------------------------
# GitHub Actions via Workload Identity Federation (no keys).
# Only this repository's main branch, in the "production" environment, can deploy.
# ---------------------------------------------------------------------------
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.this]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"        = "assertion.sub"
    "attribute.repository"  = "assertion.repository"
    "attribute.ref"         = "assertion.ref"
    "attribute.environment" = "assertion.environment"
  }

  attribute_condition = <<-EOT
    assertion.repository == "${var.github_repository}" &&
    assertion.ref == "refs/heads/main" &&
    assertion.environment == "production"
  EOT

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deploy" {
  account_id   = "terraform-deploy"
  display_name = "Terraform deploy (GitHub Actions)"
}

resource "google_service_account_iam_member" "deploy_wif" {
  service_account_id = google_service_account.deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principal://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/subject/repo:${var.github_repository}:environment:production"
}

# Roles the deploy identity needs to manage the resources in this module.
locals {
  deploy_project_roles = [
    "roles/storage.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.workloadIdentityPoolAdmin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/serviceusage.serviceUsageAdmin",
  ]
}

resource "google_project_iam_member" "deploy" {
  for_each = toset(local.deploy_project_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.deploy.email}"
}

# Note: the budget also needs roles/billing.costsManager on the billing
# account for the deploy identity. Granted once by the owner (see README).
