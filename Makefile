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
	python flows/prefect_flow.py

extract-flow:
	python flows/_00_extract_flow.py

train-promote-flow:
	python flows/_01_train_promote_flow.py

evidently-flow:
	python flows/evidently_pipelines/evidently_create_baseline.py
	python flows/evidently_pipelines/evidently_monitor.py

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

prefect-pool:
	prefect work-pool create -t process passcompass-pool
	prefect worker start --pool passcompass-pool




