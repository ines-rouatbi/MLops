from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from datetime import timedelta
from prefect.orion.schemas.schedules import IntervalSchedule
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def prepare_data():
    logger.info("📂 Preparing Data...")
    return {"message": "Data prepared successfully!"}

@task
def train_model(data):
    logger.info(f"🛠 Training Model with data: {data}")
    return {"accuracy": 95.2, "status": "Model trained successfully!"}

@task
def validate_model(model_metrics):
    logger.info(f"✅ Validating Model: Accuracy = {model_metrics['accuracy']}%")
    return "Model validation successful!"

@flow(task_runner=SequentialTaskRunner)
def ml_pipeline():
    logger.info("🚀 Starting ML Pipeline...")

    data = prepare_data()
    model_metrics = train_model(data)
    validation_status = validate_model(model_metrics)

    logger.info(f"🏁 Pipeline Completed: {validation_status}")

# Schedule the flow to run every 10 minutes
schedule = IntervalSchedule(interval=timedelta(minutes=10))
ml_pipeline.with_options(schedule=schedule)

if __name__ == "__main__":
    ml_pipeline()
