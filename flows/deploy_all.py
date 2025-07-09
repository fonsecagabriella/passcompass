# flows/deploy_all.py  – works with Prefect ≥ 2.14
from _00_extract_flow import extract_flow  # function
from _01_train_promote_flow import monthly_flow as train_promote_flow
from evidently_pipeline.evidently_create_baseline import baseline_flow as create_evidently_baseline
from evidently_pipeline.evidently_monitor import monitor_flow as evidently_monitor_flow
from master_pipeline import master_pipeline

POOL = "passcompass-process"


# 1. master
master_pipeline.deploy(
    name="scheduled_master_pipeline",
    parameters={"run_baseline": False},
    cron="30 23 * * 0",  # sunday 23:30 UTC weekly
    work_pool_name=POOL,
)

# 2. extract
extract_flow.deploy(name="production", work_pool_name=POOL)

# 3. train + promote
train_promote_flow.deploy(name="production", work_pool_name=POOL)

# 4. Evidently baseline
create_evidently_baseline.deploy(name="production", work_pool_name=POOL)

# 5. Evidently monitor
evidently_monitor_flow.deploy(name="production", work_pool_name=POOL)
