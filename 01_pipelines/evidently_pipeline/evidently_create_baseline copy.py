# flows/evidently_pipeline/evidently_create_baseline.py (Evidently 0.7+)
from evidently import Report
from evidently.metrics.data_drift   import DataDriftPreset      # ← new path
from evidently.metrics.target_drift import TargetDriftMetric    # ← new path
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------------
REFERENCE_PATH = "data/passcompass/2025_06_10/train.parquet"
OUT_DIR        = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. load reference data ------------------------------------------
ref_df = pd.read_parquet(REFERENCE_PATH)

# 2. build report with 2 metrics ----------------------------------
report = Report(
    metrics=[
        DataDriftPreset(),
        TargetDriftMetric(target_column="pass")  # pass = 0/1 column
    ]
)

# run with reference against itself (baseline)
report.run(reference_data=ref_df, current_data=ref_df)

# 3. persist both JSON (for code) and HTML (for eyeballs) ---------
report.save_json(OUT_DIR / "evidently_baseline.json")
report.save_html(OUT_DIR / "evidently_baseline.html")

print("✅ Evidently baseline saved in", OUT_DIR)