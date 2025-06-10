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

# ---- WEB APP ----
webapp-dev:
	FLASK_APP=webapp/app.py flask run --reload --port 8000

webapp-prod:
	gunicorn -w 4 -b 0.0.0.0:8000 webapp.app:app

# ---- PREFECT ----
prefect-ui:
	prefect server start

prefect-dash:
	prefect dashboard