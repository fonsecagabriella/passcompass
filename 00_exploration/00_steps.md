# PassCompass 🧭 ✅ - Exploratory phase

In this stage of the project I will get familiar with the dataset and run a first model.

## Create a virtual environment

With Conda:

```bash
conda env create -f environment.yml
conda activate passcompass-ml
```

If you prefer using pipenv, check [requirements.txt](./requirements.txt)





prefect deploy \
  01_pipelines/training_pipeline/train_logreg_flow.py:train_logreg_flow \
  --name dev \
  --param acc_min=0.80 \
  --pool default
  

  prefect deploy \
  01_pipelines/training_pipeline/train_logreg_flow.py:train_logreg_flow \
  --name dev \
  --param acc_min=0.80 \
  --pool default \
  --work-dir .