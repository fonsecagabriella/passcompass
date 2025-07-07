# 01_pipelines/promotion_pipeline/promote_best_flow.py
# ─────────────────────────────────────────────────────────────
# Promote the best MLflow run to STAGING and push its
# artifacts to Google Cloud Storage in a folder layout that
# MLflow can load directly (no zip).
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import shutil
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task, get_run_logger
from google.cloud import storage

# ───────────────────────── Configuration ──────────────────────────
BUCKET_NAME = os.getenv("MODEL_BUCKET", "passcompass-ml-bucket")
PREFIX      = "model"          # gs://bucket/model/…
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "passcompass_mlops")

# ─────────────────────── Helper functions ─────────────────────────
def _find_model_subdir(client: MlflowClient, run_id: str) -> str:
    """
    Walk the artifact tree until we find an 'MLmodel' file;
    return its parent directory path (relative to artifact root).
    """
    stack = [("", client.list_artifacts(run_id))]
    while stack:
        _, items = stack.pop()
        for it in items:
            if it.is_dir:
                stack.append((it.path, client.list_artifacts(run_id, it.path)))
            elif it.path.endswith("MLmodel"):
                return str(Path(it.path).parent)
    raise FileNotFoundError(f"No MLflow model directory found in run {run_id}")


def _upload_dir_to_gcs(local_dir: str, bucket: str, prefix: str) -> None:
    """
    Recursively upload every file in `local_dir` to
    gs://{bucket}/{prefix}/<relative_path>.
    """
    client = storage.Client()
    bucket_ref = client.bucket(bucket)

    for fp in Path(local_dir).rglob("*"):
        if fp.is_file():
            rel_path = fp.relative_to(local_dir).as_posix()
            blob = bucket_ref.blob(f"{prefix}/{rel_path}")
            blob.upload_from_filename(fp)

# ──────────────────────────── Tasks ───────────────────────────────
@task
def pick_and_register_best(
    experiment: str,
    metric: str,
    higher_is_better: bool,
    model_name: Optional[str],
) -> tuple[int, str, str]:
    """
    • Pick best run by `metric`
    • Register model → STAGING
    • Return (version, run_id, resolved_model_name)
    """
    log = get_run_logger()
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment)
    if not exp:
        raise ValueError(f"Experiment '{experiment}' not found")

    runs = client.search_runs([exp.experiment_id])
    if not runs:
        raise ValueError("No runs in experiment")

    key = metric
    best_run = (
        max(runs, key=lambda r: r.data.metrics.get(key, float("-inf")))
        if higher_is_better
        else min(runs, key=lambda r: r.data.metrics.get(key, float("inf")))
    )
    best_val = best_run.data.metrics.get(key)
    log.info(f"Best run {best_run.info.run_id}: {key} = {best_val}")

    if not model_name:
        t = best_run.data.tags.get("model_type", "generic")
        model_name = f"passcompass_{t}"

    mv = mlflow.register_model(f"runs:/{best_run.info.run_id}/model", model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    log.info(f"Model {model_name} v{mv.version} promoted to STAGING")
    return mv.version, best_run.info.run_id, model_name


@task
def upload_artifacts_to_gcs(
    run_id: str,
    model_name: str,
    model_version: int,
    bucket_name: str = BUCKET_NAME,
    prefix: str = PREFIX,
):
    """
    • Locate model dir in run artifacts
    • Download it
    • Upload directory contents to GCS
    """
    log = get_run_logger()
    client = MlflowClient()

    subdir = _find_model_subdir(client, run_id)
    log.info(f"Found model directory '{subdir}'")

    # Download to temp dir
    tmp_dir = tempfile.mkdtemp()
    local_model_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/{subdir}",
        dst_path=tmp_dir,
    )

    # Upload directory tree
    gcs_prefix = f"{prefix}/{model_name}_v{model_version}"
    _upload_dir_to_gcs(local_model_dir, bucket_name, gcs_prefix)
    log.info(f"📤 Uploaded to gs://{bucket_name}/{gcs_prefix}/")

    shutil.rmtree(tmp_dir, ignore_errors=True)

# ───────────────────────────── Flow ────────────────────────────────
@flow(name="promote_best_model")
def promote_best_model_flow_gcs(
    experiment: str = "passcompass_mlops",
    metric: str = "val_macro_avg_f1-score",
    higher_is_better: bool = True,
    model_name: Optional[str] = None,
):
    """
    Promote the best run & push its artifact directory to GCS
    (raw, unzipped layout).
    """
    ver, run_id, name = pick_and_register_best(
        experiment, metric, higher_is_better, model_name
    )
    upload_artifacts_to_gcs(run_id, name, ver)
    get_run_logger().info(f"✅ Promotion complete — version {ver}")

# ────────────────────────── CLI entry ──────────────────────────────
if __name__ == "__main__":
    promote_best_model_flow(experiment=EXPERIMENT)
