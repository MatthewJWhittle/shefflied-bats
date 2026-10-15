variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "Region for buckets."
  type        = string
  default     = "europe-west2"
}

variable "bucket_name" {
  description = "Name of the data bucket (inputs/, runs/, published/)."
  type        = string
}

variable "state_bucket_name" {
  description = "Name of the Terraform state bucket."
  type        = string
}

variable "billing_account" {
  description = "Billing account ID used for the budget alert."
  type        = string
}

variable "budget_amount" {
  description = "Monthly budget in the billing account currency."
  type        = number
  default     = 1
}

variable "budget_currency" {
  description = "Currency code of the billing account."
  type        = string
  default     = "GBP"
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, as owner/name."
  type        = string
}

variable "model_run_impersonator_email" {
  description = "User allowed to impersonate the model-run service account (token creator)."
  type        = string
}
