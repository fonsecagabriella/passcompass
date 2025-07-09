# flows/deploy_master.py

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from flows.master_pipeline import master_pipeline

deployment = Deployment.build_from_flow(
    flow=master_pipeline,
    name="scheduled_master_pipeline",
    parameters={"run_baseline": False},
    schedule=(CronSchedule(cron="0 8 * * *")),  # Every day at 8AM UTC
    work_queue_name="default",  # Adjust this to match your Prefect queue name
)

if __name__ == "__main__":
    deployment.apply()
