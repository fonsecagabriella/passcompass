"""
Create Evidently baseline report
Evidently 0.7.x  •  Prefect 3.x
"""

from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift
from prefect import flow, task

# -------------------------------------------------------------------
REFERENCE_PATH = Path("data/passcompass/2025_06_10/train.parquet")
OUT_DIR        = Path("reports")
TARGET_COL     = "pass"                   #  ❰ change if your flag differs
# -------------------------------------------------------------------


@task(name="Load reference data")
def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL!r} not found in {path}")
    return df


@task(name="Build Evidently report")
def build_report(ref: pd.DataFrame) -> Report:
    report = Report(metrics=[
        DataDriftPreset(),
        ValueDrift(column=TARGET_COL)     # ← fixed keyword
    ])
    report.run(reference_data=ref, current_data=ref)
    return report


@task(name="Persist report")
def save_report(report: Report, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_json(out_dir / "evidently_baseline.json")
    report.save_html(out_dir / "evidently_baseline.html")
    print("✅ baseline saved in", out_dir.resolve())


@flow(name="create_evidently_baseline")
def baseline_flow(
    ref_path: Path = REFERENCE_PATH,
    out_dir: Path = OUT_DIR,
):
    ref = load_reference(ref_path)
    rpt = build_report(ref)
    save_report(rpt, out_dir)


if __name__ == "__main__":
    # run once from CLI
    baseline_flow()
    # or deploy with Prefect serve / prefect deploy if you wish
