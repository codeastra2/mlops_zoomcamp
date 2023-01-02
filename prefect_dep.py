from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta
from giba_xgb.flows import main

deployment = Deployment.build_from_flow(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    work_queue_name="ml"
)

deployment.apply()
