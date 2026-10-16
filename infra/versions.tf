terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Remote state. The bucket name is supplied at init time:
  #   terraform init -backend-config="bucket=<state-bucket-name>"
  # See infra/README.md for the one-off bootstrap.
  backend "gcs" {
    prefix = "terraform/state"
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}
