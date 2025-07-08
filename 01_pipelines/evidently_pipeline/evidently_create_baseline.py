"""
Create Evidently baseline report.

Running locally or on GCP
=========================
* **Local** (default): generates Evidently HTML/JSON under ``reports/``.
* **GCS** (set ``ENVIRONMENT=gcs``): also uploads the *reference* parquet to
  ``gs://<bucket>/reference/<file>``.

Notes
-----
* Reference‑data path is now resolved *absolutely* so that the flow can be
  launched from any working directory.
* A clearer error is raised when the parquet file is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift  # target‑drift metric
from evidently.presets import DataDriftPreset
from google.cloud import storage
from prefect import flow, task

# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/passcompass
REFERENCE_PATH = PROJECT_ROOT / "data" / "passcompass" / "2025_06_10" / "train.parquet"
TARGET_COL = "pass"  # 0/1 label
OUT_DIR = PROJECT_ROOT / "reports"
# ────────────────────────────────────────────────────────────────


@task
def load_reference(path: Path) -> pd.DataFrame:
    """Read the reference parquet file and validate it contains the target column."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Reference parquet not found at {path}. "
            "Have you generated it with the training pipeline?"
        )

    df = pd.read_parquet(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL!r} column not found in {path}")
    return df


@task
def build_snapshot(ref: pd.DataFrame):
    """Run Evidently and return the *snapshot* object produced by ``Report.run``.

    In Evidently ≥0.7 the `Report` itself is just a *template*. Once you call
    ``run`` you get a *snapshot* that contains the computed results **and** the
    convenience exporters like ``save_html``/``save_json``/``dict``.
    """

    report = Report(metrics=[DataDriftPreset(), ValueDrift(column=TARGET_COL)])
    snapshot = report.run(reference_data=ref, current_data=ref)
    return snapshot


@task
def save_report(snapshot, out_dir: Path):
    """Persist the Evidently snapshot locally as HTML and JSON.

    For Evidently ≥0.7 the snapshot supports ``save_html`` and ``save_json``.
    We keep the same graceful‑degradation logic in case of unexpected API
    changes.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON when possible
    json_path = out_dir / "evidently_baseline.json"
    if hasattr(snapshot, "save_json"):
        snapshot.save_json(json_path)
    else:
        try:
            json_path.write_text(snapshot.json(), encoding="utf-8")  # type: ignore[attr-defined]
        except AttributeError:
            print("⚠️  JSON export unavailable – skipping.")

    # Save HTML when possible
    html_path = out_dir / "evidently_baseline.html"
    if hasattr(snapshot, "save_html"):
        snapshot.save_html(html_path)
    else:
        try:
            html_content = snapshot.html()  # type: ignore[attr-defined]
            html_path.write_text(html_content, encoding="utf-8")
        except AttributeError:
            print("⚠️  HTML export unavailable – skipping.")

    print("✅ Baseline saved to", out_dir.resolve())


@task
def upload_to_gcs(path: Path, bucket_name: str, destination: str | None = None):
    """Upload *path* to *gs://bucket_name/destination*.

    Google credentials must be configured via ``GOOGLE_APPLICATION_CREDENTIALS``
    or similar mechanism.
    """

    destination = destination or path.name
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_filename(str(path))
    print(f"✅ Uploaded {path.name} → gs://{bucket_name}/{destination}")


@flow(name="create_evidently_baseline")
def baseline_flow(
    ref_path: Path = REFERENCE_PATH,
    out_dir: Path = OUT_DIR,
    bucket_name: str | None = os.getenv("GCS_BUCKET"),
):
    """Generate a baseline Evidently report and optionally push data to GCS.

    Returns the Evidently *snapshot* so the caller can inspect it in notebooks
    (e.g. ``snapshot.dict()``).
    """

    ref = load_reference(ref_path)
    snapshot = build_snapshot(ref)
    save_report(snapshot, out_dir)

    # if ENVIRONMENT == "gcs" and bucket_name:
    if os.getenv("ENVIRONMENT", "local").lower() == "gcs" and bucket_name:

        upload_to_gcs(ref_path, bucket_name, f"evidently/reference/{ref_path.name}")
        upload_to_gcs(
            out_dir / "evidently_baseline.html",
            bucket_name,
            "evidently/reports/evidently_baseline.html",
        )
        upload_to_gcs(
            out_dir / "evidently_baseline.json",
            bucket_name,
            "evidently/reports/evidently_baseline.json",
        )

        print(
            f"✅ Reference parquet uploaded to gs://{bucket_name}/evidently/reference/{ref_path.name}"
        )

    return snapshot


if __name__ == "__main__":
    baseline_flow()
