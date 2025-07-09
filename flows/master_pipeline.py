# master_pipeline.py

from datetime import datetime

from prefect import flow
from prefect.deployments import run_deployment


@flow(name="master_pipeline")
def master_pipeline(run_baseline: bool = False):
    """
    Orchestrates the full ML pipeline:
    - Extract & clean data
    - Train and promote best model
    - Run Evidently baseline (optional)
    - Run Evidently monitor
    """

    print(f"🚀 Starting ML pipeline at {datetime.utcnow()}")

    # Step 1: Extract and clean data
    print("📥 Running extract flow...")
    run_deployment(name="00_extract_flow/production")

    # Step 2: Train and promote best model
    print("🤖 Running training and promotion flow...")
    run_deployment(name="01_train_promote_flow/production")

    # Step 3: Run baseline report if flag is set
    if run_baseline:
        print("📊 Running Evidently baseline...")
        run_deployment(name="evidently_create_baseline/production")

    # Step 4: Monitor for drift
    print("🧭 Running Evidently drift monitoring...")
    run_deployment(name="evidently_monitor/production")

    print(f"✅ ML pipeline finished at {datetime.utcnow()}")


if __name__ == "__main__":
    # Set environment variable for GCS bucket if needed
    # gcs_bucket = String.load("gcs_bucket_name")
    # if gcs_bucket:
    #    print(f"Using GCS bucket: {gcs_bucket.value}")

    # Run the master pipeline with baseline report enabled
    master_pipeline(run_baseline=True)
