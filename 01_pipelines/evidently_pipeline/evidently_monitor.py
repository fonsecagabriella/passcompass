# flows/evidently_pipeline/evidently_monitor.py
from prefect import flow, task
from pathlib import Path
from evidently.report import Report
from evidently.metrics.data_drift import DataDriftPreset
from evidently.metrics.target_drift import TargetDriftMetric   # 👈 new path
import pandas as pd

ref = pd.read_parquet("data/.../train.parquet")


BASELINE_ARTIFACT = "reports/evidently_baseline.json"
RAW_DIR   = "data/passcompass"            # where extract_flow puts new t-stamps
OUT_DIR   = Path("reports/monitor")

@task
def latest_scoring_batch() -> pd.DataFrame:
    latest = max(Path(RAW_DIR).glob("20*_*/students_clean.parquet"))
    return pd.read_parquet(latest)

@task
def run_evidently(current: pd.DataFrame):
    ref_report = Report.from_json(Path(BASELINE_ARTIFACT).read_text())
    reference  = ref_report._data["reference_data"]          # quick hack
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=pd.DataFrame(reference), current_data=current)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUT_DIR / f"monitor_{pd.Timestamp.utcnow():%Y_%m_%d}.html"
    report.save_html(fname)
    return fname

@flow(name="evidently_monitor_flow")
def monitor_flow():
    cur = latest_scoring_batch()
    html_path = run_evidently(cur)
    print("✅ Evidently report saved to", html_path)

if __name__ == "__main__":
    monitor_flow.serve(                      # Prefect 2.x deployment
        name="monitor-monthly",
        cron="0 7 1 * *",                   # 07:00 UTC on the 1st monthly
        tags={"project":"passcompass","type":"monitor"},
    )
