"""
Register the best model from an experiment run.

Usage:
    python register_best.py --experiment passcompass_hyperopt_rf \
                            --metric val_accuracy \
                            --higher_is_better
"""
import argparse, mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True)
parser.add_argument("--metric", required=True)
parser.add_argument("--higher_is_better", action="store_true")
parser.add_argument("--model-name", default="passcompass_students")
args = parser.parse_args()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 1) find all runs in the experiment
exp = mlflow.get_experiment_by_name(args.experiment)
print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
runs = mlflow.search_runs(exp.experiment_id, output_format="list")

# 2) pick best by metric
best_run = max(
    runs,
    key=lambda r: float(r.data.metrics.get(args.metric, -1e9))
) if args.higher_is_better else min(
    runs,
    key=lambda r: float(r.data.metrics.get(args.metric, 1e9))
)

model_uri = f"runs:/{best_run.info.run_id}/model"
print("Registering model from:", model_uri)

# 3) register & transition
result = mlflow.register_model(model_uri, args.model_name)
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name=args.model_name,
    version=result.version,
    stage="Staging",
    archive_existing_versions=True
)

print(f"Model v{result.version} set to STAGING.")
