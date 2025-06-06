# ====== ENV ======
env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml

# ====== MLFLOW ======
mlflow-ui:
	mlflow ui --config-file conf/mlflow_local.yaml

# ====== FLOW ======
run-flow:
	python 01_pipeline/prefect_flow.py

train-only:
	python 01_pipeline/train.py
