"""
Monitor new scoring batch against baseline
Evidently 0.7.x  •  Prefect 3.x
"""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
from prefect import flow, task

# ─── paths & constants ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "passcompass"
BASELINE_JSON = PROJECT_ROOT / "reports" / "evidently_baseline.json"
OUT_PARENT_DIR = PROJECT_ROOT / "reports" / "monitor"
TARGET_COL = "pass"
MIN_ROWS = 50  # skip very small batches
# ──────────────────────────────────────────────────────────────


@task
def find_latest_batch() -> Path:
    """Return path to newest students_clean.parquet in DATA_DIR/**/"""
    batches = sorted(DATA_DIR.glob("*/students_clean.parquet"))
    if not batches:
        raise FileNotFoundError("No scoring batches found in data dir")
    return batches[-1]  # newest by lexicographic YYYY_MM_DD


@task
def load_current(batch_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(batch_path)
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Batch too small ({len(df)} rows). " f"Need at least {MIN_ROWS} for drift tests."
        )
    return df


@task
def run_evidently(current: pd.DataFrame) -> Report:
    """
    Compare the current batch to the stored baseline batch with Evidently.
    Works with baselines saved by Evidently ≤0.7.x or ≥0.8.
    """
    # ---------- load baseline file ----------
    # baseline_dict = json.loads(BASELINE_JSON.read_text())

    # ---------- extract reference rows no matter the schema ----------
    # if "reference_data" in baseline_dict:  # Evidently ≤0.7.x
    #     ref_rows = baseline_dict["reference_data"]
    # else:  # Evidently ≥0.8
    #     ref_rows = baseline_dict["data"]["reference"]["data"]

    # reference_data = pd.DataFrame(ref_rows)

    reference_data = pd.read_parquet(PROJECT_ROOT / "data/passcompass/2025_06_10/train.parquet")
    # ---------- run drift report ----------
    report = Report(
        metrics=[
            DataDriftPreset(),
            ValueDrift(column=TARGET_COL),
        ]
    )
    my_eval = report.run(reference_data=reference_data, current_data=current)
    return my_eval


@task
def persist_report(report: Report, batch_path: Path):
    stamp = batch_path.parent.name  # e.g. 2025_07_15
    out_dir = OUT_PARENT_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"monitor_{stamp}.html"
    json_path = out_dir / f"monitor_{stamp}.json"

    # report.save_html(html_path)
    # report.save_json(json_path)

    json_path.write_text(report.json(), encoding="utf-8")

    print("✅ Drift report saved →", html_path.relative_to(PROJECT_ROOT))
    return json_path


@flow(name="evidently_monitor_flow")
def monitor_flow():
    latest_batch = find_latest_batch()
    cur_df = load_current(latest_batch)
    rpt = run_evidently(cur_df)
    persist_report(rpt, latest_batch)


if __name__ == "__main__":
    # 1️⃣ local test: just run python evidently_monitor.py
    monitor_flow()
    # 2️⃣ after you confirm it works, deploy:
    # prefect deploy -n evidently-monitor -f evidently_monitor.py
