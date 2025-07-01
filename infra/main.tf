provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "model_bucket" {
  name     = var.bucket_name
  location = "EU"
  force_destroy = true

  uniform_bucket_level_access = true
}

resource "google_storage_bucket_object" "model_folder" {
  name   = "model/.keep"
  bucket = google_storage_bucket.model_bucket.name
  source = "${path.module}/dummy/.keep"
}

resource "google_storage_bucket_object" "data_folder" {
  name   = "data/.keep"
  bucket = google_storage_bucket.model_bucket.name
  source = "${path.module}/dummy/.keep"
}
