# 🛠️ Project Setup Guide

This guide outlines how to set up and run the project from scratch. The setup includes provisioning cloud infrastructure with Terraform, deploying the model API, and preparing for end-to-end machine learning operations.


## Folder structure

PASSCOMPASS/
│
├── 00_exploration/              # existing notebooks
|                                # raw + processed CSVs
│
├── **01_pipelines/**             # code that Prefect will call
│   ├── **train.py**             # train + log + register model
│   └── **prefect_flow.py**      # orchestration wrapper
│
├── **conf/**
│   └── **mlflow_local.yaml**    # central tracking config
│
├── **Makefile**                 # convenience targets
├── **environment.yml**          # include mlflow & prefect
├── **.env.example**             # env vars (ignored in .gitignore)
│
├── LICENSE
└── README.md

```bash
make env-create
make mlflow-ui          # open http://127.0.0.1:5000
make run-flow           # kicks off the Prefect pipeline
```


---

## 📦 Prerequisites

Make sure you have the following installed:

- [Python 3.8+](https://www.python.org/)
- [Terraform](https://developer.hashicorp.com/terraform/downloads)
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk)
- Docker
- A Google Cloud Project (you will need the **project ID**)

---

## Notes
There are two options to run this project:
1. Local
2. Cloud (partially) 

You can set a local variable ENVIRONMENT as 

1. `local`
2. `gcs` (for google cloud console)

The Cloud setup is currently partial. This has been done intentionally due to the (time & budget) limitation of this project.
Data for traing is stored locally. In the cloud option, you can deploy the best model to the cloud in a bucket. Evidently reports are also stored on the bucket. 
Terraform is used to create the bucket.

--- 

## Instructions

### 0.0 Create a virtual machine
``` bash
make env-create
```

### 0.1 Set up local variables

``` bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
MLFLOW_ARTIFACT_ROOT=./artifacts
PREFECT_API_URL=http://127.0.0.1:4200/api          # optional
PREFECT_SEND_ANONYMOUS_TELEMETRY=0
PYTHONPATH=${PYTHONPATH}:./src
MODEL_NAME=passcompass_generic
MODEL_STAGE=Staging 
MLFLOW_EXPERIMENT=passcompass_mlops
PREFECT__FLOWS__EXECUTION__DEFAULT_GIT_DESTINATION="~/prefect/git-storage" # edit path
ENVIRONMENT=local # local or gcs
GCS_BUCKET=gs://passcompass-ml-bucket # edit path
GCS_MODEL_URI=gs://passcompass-ml-bucket/model/passcompass_generic_v16 # edit path
GCS_DATA_URI=gs://passcompass-ml-bucket/data # edit path
LOCAL_DATA_URI=passcompass/data/passcompass # edit path
```


### 0.2 **(if `ENVIRONMENT=gcs`)** Set up Google Cloud Project 

1. Create a GCP project if you haven’t already.
2. Enable the following APIs:
   - Cloud Run
   - Cloud Storage
   - Artifact Registry
3. Authenticate with GCP CLI:
   ```bash
   gcloud auth application-default login
   ```

### 0.2.1 Provision Infrastructure with Terraform

1. Navigate to the `infra/` directory: `cd infra`

2. Create a `terraform.tfvars` file to pass in your project ID:

3. Initiliase and apply terraform

```bash
terraform init
terraform apply -var-file="terraform.tfvars"
```

**✅ What this step does**
- Creates a GCS bucket: passcompass-ml-bucket
- Simulates two folder paths by adding empty .keep files:
    - model/
    - data/

### 0.3 Run MLflow & Prefect Server locally

```bash
make prefect-ui
make prefect-dashboard
make mlflow-ui

```

### 1.0 Run Prefect flows

1. Extract data
`make extract-flow`
Ps: You can comment out lines of the main function to register a CRON deployment. 

2. Train and promote model
`make train-promote-flow`

3. Run web application and try a prediction
`make webapp-dev`

You can navigate to:

[add screenshot of web app]

4. Draft Evidently flow simple one




### Tests
Run tests anytime with
`pytest -q`

**Currently implemented tests:**
- Does the data-download task leave a file where you expect?
- Does the model get one basic prediction right?

For the future:
- Does mlflow return a model?
- Does the “promote best model” helper pick the right run?
- Can the Flask endpoint answer a happy-path request?
- Is the monitoring flow with Evidently at least able to run end-to-end?