# flows/promotion_pipeline/promote_best_flow.py
import os

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "passcompass_mlops")
ALIAS = os.getenv("MODEL_ALIAS", "best_202506")
print(f"Using experiment: {EXPERIMENT}")


@task
def pick_and_register_best(
    experiment: str,
    metric: str,
    higher_is_better: bool,
    model_name: str | None,
    mlflow_uri: str,
    alias: ALIAS,
) -> int | None:
    log = get_run_logger()

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        log.error(f"Experiment '{experiment}' not found.")
        return None

    print(f"Using experiment: {exp.name} (ID: {exp.experiment_id})")

    runs = client.search_runs(experiment_ids=[exp.experiment_id])

    if not runs:
        log.warning("No runs found.")
        return None

    key = metric
    best_run = (
        max(runs, key=lambda r: r.data.metrics.get(key, float("-inf")))
        if higher_is_better
        else min(runs, key=lambda r: r.data.metrics.get(key, float("inf")))
    )

    best_val = best_run.data.metrics.get(key)
    log.info(f"Best run {best_run.info.run_id}: {key} = {best_val}")

    # ── Derive model registry name if none supplied ──────────────────────
    if not model_name:
        mtype = best_run.data.tags.get("model_type", "generic")
        model_name = f"passcompass_{mtype}"

    model_uri = f"runs:/{best_run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version,
    )
    log.info(f"Model {model_name} v{mv.version} promoted to STAGING.")
    return mv.version


@flow(name="promote_best_model")
def promote_best_model_flow_local(
    experiment: str = "passcompass",
    metric: str = "val_macro_avg_f1-score",
    higher_is_better: bool = True,
    model_name: str | None = None,  # ← now optional
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
):
    """
    Promote the best run (by `metric`) to MLflow Model Registry / STAGING.
    If `model_name` is omitted, it derives one from the run's `model_type` tag.
    """
    ver = pick_and_register_best(experiment, metric, higher_is_better, model_name, mlflow_uri)
    if ver:
        get_run_logger().info(f"✅ Promotion complete – version {ver}")
    else:
        get_run_logger().warning("⚠️  Promotion skipped – nothing to promote")


if __name__ == "__main__":
    promote_best_model_flow_local(experiment=EXPERIMENT)
