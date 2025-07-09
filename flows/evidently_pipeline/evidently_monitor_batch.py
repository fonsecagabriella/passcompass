"""
Evidently **batch drift monitoring** flow
========================================

Simple, beginner‑friendly Prefect 3 flow that compares a *current* production
slice against the *reference* (training) dataset and stores an Evidently drift
report locally **and** (optionally) in Google Cloud Storage.

Run ad‑hoc::

    python evidently_pipeline/evidently_monitor_batch.py \
        --current-path data/passcompass/2025_07_02/pred_input.parquet

Run with GCS upload::

    ENVIRONMENT=gcs \
    GCS_EVIDENTLY_URI=gs://passcompass-ml-bucket/evidently \
    python evidently_pipeline/evidently_monitor_batch.py \
        --current-path data/passcompass/2025_07_02/pred_input.parquet

Schedule daily via Prefect in a notebook or separate script::

    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule
    from evidently_pipeline.evidently_monitor_batch import monitor_flow

    Deployment.build_from_flow(
        flow=monitor_flow,
        name="daily-drift",
        parameters={"current_path": "data/passcompass/$(date +%F)/pred_input.parquet"},
        schedule=CronSchedule(cron="0 3 * * *", timezone="Europe/Amsterdam"),
    ).apply()

Requires Evidently ≥ 0.7.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from google.cloud import storage
from prefect import flow, task

# ────────────────────────────────────────────────────────────────
# Configuration – tweak paths as needed
# ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / "data" / "passcompass" / "2025_06_10" / "train.parquet"
TARGET_COL = "pass"  # set to None if you don't have ground‑truth yet
REPORTS_ROOT = PROJECT_ROOT / "reports" / "drift"

# ────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket_name, prefix) for ``<bucket>`` or ``gs://bucket/prefix``."""
    if uri.startswith("gs://"):
        bucket, *rest = uri[5:].split("/", 1)
        prefix = rest[0] if rest else ""
    else:
        bucket, prefix = uri, ""
    return bucket, prefix.rstrip("/")


# ────────────────────────────────────────────────────────────────
# Tasks
# ────────────────────────────────────────────────────────────────


@task
def load_parquet(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


@task
def build_snapshot(ref: pd.DataFrame, cur: pd.DataFrame):
    """Return the *snapshot* object produced by `Report.run()` (contains exporters)."""
    metrics = [DataDriftPreset()]
    if TARGET_COL in ref.columns and TARGET_COL in cur.columns:
        metrics.append(ValueDrift(column=TARGET_COL))

    rpt = Report(metrics=metrics)
    snapshot = rpt.run(reference_data=ref, current_data=cur)
    return snapshot  # this has save_html / save_json etc.


@task
def save_report(snapshot, out_dir: Path) -> tuple[Path | None, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_p = out_dir / "drift_report.html"
    json_p = out_dir / "drift_report.json"
    html_ok = json_ok = False

    # ---------- HTML ----------
    try:
        if hasattr(snapshot, "save_html"):
            # try the file-writing path first
            snapshot.save_html(html_p)
        if not html_p.exists():  # if still missing …
            html_str = snapshot.save_html()  # … get the HTML string …
            html_p.write_text(html_str, encoding="utf-8")  # … and save it ourselves
        html_ok = html_p.exists()
    except Exception as exc:
        print(f"⚠️  HTML export failed: {exc}")

    # ---------- JSON ----------
    try:
        if hasattr(snapshot, "save_json"):
            snapshot.save_json(json_p)  # might be a no-op
        if not json_p.exists():  # fallback
            if hasattr(snapshot, "as_dict"):
                import json as _json

                json_p.write_text(
                    _json.dumps(snapshot.as_dict(), default=str, indent=2), encoding="utf-8"  # type: ignore[attr-defined]
                )
            else:
                json_p.write_text(snapshot.json(), encoding="utf-8")  # type: ignore[attr-defined]
        json_ok = json_p.exists()
    except Exception as exc:
        print(f"⚠️  JSON export failed: {exc}")

    if html_ok or json_ok:
        print("✅ Drift report saved to", out_dir.resolve())
    else:
        print("❌ Export produced no files – check Evidently version.")

    return (html_p if html_ok else None, json_p if json_ok else None)


@task
def upload_to_gcs(local_path: Path, bucket_uri: str, destination: str | None = None):
    path = Path(local_path)
    bucket_id, base_prefix = _split_gs_uri(bucket_uri)
    destination = destination or path.name
    blob_name = "/".join(filter(None, (base_prefix, destination)))

    client = storage.Client()
    bucket = client.bucket(bucket_id)
    if not bucket.exists():
        raise ValueError(f"Bucket '{bucket_id}' does not exist – create it first.")

    bucket.blob(blob_name).upload_from_filename(str(path))
    print(f"✅ Uploaded {path.name} → gs://{bucket_id}/{blob_name}")


# ────────────────────────────────────────────────────────────────
# Flow
# ────────────────────────────────────────────────────────────────


@flow(name="monitor_drift_batch")
def monitor_flow(
    current_path: Path,
    reference_path: Path = REFERENCE_PATH,
    reports_root: Path = REPORTS_ROOT,
    gcs_uri: str | None = os.getenv("GCS_EVIDENTLY_URI") or os.getenv("GCS_BUCKET"),
):
    """Compare *current_path* with reference and persist drift report."""

    ref_df = load_parquet(reference_path)
    cur_df = load_parquet(current_path)
    snapshot = build_snapshot(ref_df, cur_df)

    today = dt.date.today().isoformat()
    out_dir = reports_root / today
    html_p, json_p = save_report(snapshot, out_dir)

    if os.getenv("ENVIRONMENT", "local").lower() == "gcs" and gcs_uri:
        if html_p and html_p.exists():
            upload_to_gcs(html_p, gcs_uri, f"evidently/reports/drift/{today}/{html_p.name}")
        if json_p and json_p.exists():
            upload_to_gcs(json_p, gcs_uri, f"evidently/reports/drift/{today}/{json_p.name}")

    return snapshot


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--current-path", type=Path, required=True, help="Parquet file with today's data slice"
    )
    args = p.parse_args()

    monitor_flow(current_path=args.current_path)
