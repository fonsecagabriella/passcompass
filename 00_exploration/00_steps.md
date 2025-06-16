# PassCompass 🧭 ✅ - Exploratory phase

In this stage of the project I will get familiar with the dataset and run a first model.

## Create a virtual environment

With Conda:

```bash
conda env create -f environment.yml
conda activate passcompass-ml
```

If you prefer using pipenv, check [requirements.txt](./requirements.txt)

- Start prefect `prefect server start`
- First time only: create work-pool `prefect work-pool create default -t process`
- Star Mlflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:./artifacts \
  --serve-artifacts \
  --host 127.0.0.1 \
  --port 5001
```

or

`mlflow server --config conf/mlflow_local.yaml --serve-artifacts`

When you're ready do deploy and/or is something changes on code:
  
```bash
prefect deploy \                                          
  01_pipelines/training_pipeline/train_logreg_flow.py:train_logreg_flow \
  --name dev \
  --param acc_min=0.8 \
  --pool default
```

to schedyle a run: `prefect deployment run 'train_logreg_flow/dev'`

