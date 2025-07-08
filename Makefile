# ====== ENV ======
env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml

# ====== MLFLOW ======
mlflow-ui:
	mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:./artifacts \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5001

# ====== FLOW ======
run-flow:
	python 01_pipelines/prefect_flow.py

extract-flow:
	python 01_pipelines/00_extract_flow.py

train-promote-flow:
	python 01_pipelines/01_train_promote_flow.py

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