output "workload_identity_provider" {
  description = "Value for the GCP_WIF_PROVIDER production environment variable."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deploy_service_account" {
  description = "Value for the GCP_DEPLOY_SA production environment variable."
  value       = google_service_account.deploy.email
}

output "run_service_account" {
  value = google_service_account.run.email
}
