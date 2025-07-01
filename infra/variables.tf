variable "project_id" {
  description = "Your GCP project ID"
  type        = string
}

variable "region" {
  default = "EUROPE-WEST4"
}

variable "bucket_name" {
  default = "passcompass-ml-bucket"
}
