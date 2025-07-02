"""
Create Evidently baseline report
Evidently 0.7.x  •  Prefect 3.x
"""

from pathlib import Path
import pandas as pd
from prefect import flow, task

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift      # target drift replacement

# ────────────────────────────────────────────────────────────────
REFERENCE_PATH = Path("data/passcompass/2025_06_10/train.parquet")
TARGET_COL     = "pass"                       # 0/1 label
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/passcompass
OUT_DIR      = PROJECT_ROOT / "reports" 
# ────────────────────────────────────────────────────────────────


@task
def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL!r} not found in {path}")
    return df


@task
def build_report(ref: pd.DataFrame) -> Report:
    report = Report(metrics=[
        DataDriftPreset(),                   # full feature drift
        ValueDrift(column=TARGET_COL)        # target-column drift
    ])
    my_report = report.run(reference_data=ref, current_data=ref)
    return my_report

@task
def save_report(report: Report, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(out_dir / "evidently_baseline.html")
    report.save_json(out_dir / "evidently_baseline.json")
    print("✅ Baseline saved to", out_dir.resolve())


@flow(name="create_evidently_baseline")
def baseline_flow(
    ref_path: Path = REFERENCE_PATH,
    out_dir: Path = OUT_DIR,
):
    ref  = load_reference(ref_path)
    rpt  = build_report(ref)  
    save_report(rpt, out_dir)


if __name__ == "__main__":
    baseline_flow()
