output "workload_identity_provider" {
  description = "Value for the GCP_WORKLOAD_IDENTITY_PROVIDER repository variable."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deploy_service_account" {
  description = "Value for the GCP_DEPLOY_SERVICE_ACCOUNT repository variable."
  value       = google_service_account.deploy.email
}

output "run_service_account" {
  value = google_service_account.run.email
}
