# 🧠 MLOps Project: End-to-End ML Pipeline

This project demonstrates a complete machine learning workflow — from infrastructure provisioning to model deployment — using modern MLOps tools and best practices.

---

## 🛠️ Project Setup Guide

This guide outlines how to set up and run the project from scratch. The setup includes provisioning cloud infrastructure with Terraform, deploying the model API, and preparing for end-to-end machine learning operations.

---

### 📦 Prerequisites

Make sure you have the following installed:

- [Python 3.8+](https://www.python.org/)
- [Terraform](https://developer.hashicorp.com/terraform/downloads)
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk)
- Docker
- A Google Cloud Project (you will need the **project ID**)

---

### 🌍 Step 1: Set up Google Cloud Project

1. Create a GCP project if you haven’t already.
2. Enable the following APIs:
   - Cloud Run
   - Cloud Storage
   - Artifact Registry
3. Authenticate with GCP CLI:
   ```bash
   gcloud auth application-default login
   ```

### 🌍 Step 2: Provision Infrastructure with Terraform

1. Navigate to the `infra/` directory: `cd infra`

2. Create a `terraform.tfvars` file to pass in your project ID:

3. Initiliase and apply terraform

```bash
terraform init
terraform apply -var-file="terraform.tfvars"
```

**✅ What This Step Does**
- Creates a GCS bucket: passcompass-ml-bucket
- Simulates two folder paths by adding empty .keep files:
    - model/
    - data/

### Create Virtual Machine
### Run mlflow and prefecto