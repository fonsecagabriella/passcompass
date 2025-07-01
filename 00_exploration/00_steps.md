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
  --name dev --pool default --param acc_min=0.8
```

to schedyle a run: `prefect deployment run 'train_logreg_flow/dev'`

**IF THERE'S CHANGE TO CODE**
- commit to github
```bash
git add 01_pipelines/training_pipeline/train_utils.py
git commit -m "Fix threshold sweep dtype bug"
git push origin main
```
- deploy like example above
- schedule a run


**DEPLOY BEST MODEL**

# build deployment
prefect deploy \
  01_pipelines/promotion_pipeline/promote_best_flow.py:promote_best_model_flow \
  --name promote_best --pool default \
  --param experiment=passcompass \
  --param metric=val_macro_avg_f1-score \
  --param higher_is_better=true       # F1 ↑ is better

# run once
prefect deployment run promote_best_model_flow/promote_best