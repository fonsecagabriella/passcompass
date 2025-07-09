import os

from dotenv import load_dotenv
from prefect import flow
from promotion_pipeline.promote_best_flow_cloud import (
    promote_best_model_flow_gcs as promote_best_flow_gcs,
)
from promotion_pipeline.promote_best_flow_local import (
    promote_best_model_flow_local as promote_best_flow_local,
)
from training_pipeline.train_gbc_flow import train_gbc_flow
from training_pipeline.train_logreg_flow import train_logreg_flow
from training_pipeline.train_randforest_flow import train_randforest_flow

# Load environment variables from a .env file into the environment
load_dotenv()


@flow(name="train_promote_flow")
def monthly_flow():
    """
    This flow trains three different models and then promotes the best one.
    It decides whether to use a local or cloud-based promotion strategy
    based on the 'ENVIRONMENT' variable in a .env file.
    """
    # These training flows will run regardless of the environment
    print("Starting training flows...")
    train_logreg_flow()
    train_randforest_flow()
    train_gbc_flow()

    # Get the environment setting from the loaded environment variables
    environment = os.getenv("ENVIRONMENT")

    # Decide which promotion function to call based on the environment
    if environment == "gcs":
        print("ENVIRONMENT is 'gcs', running cloud promotion flow. ☁️")
        promote_best_flow_gcs(
            experiment=os.getenv(
                "MLFLOW_EXPERIMENT"
            ),  # You can override the experiment name here if needed
            metric="val_macro_avg_f1-score",  # Override metric if needed
            higher_is_better=True,
            model_name=os.getenv(
                "MODEL_NAME", "passcompass_model"
            ),  # Override model name if needed
            mlflow_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"
            ),  # Pass the URI if it's dynamic
        )
    else:
        # Default to the local flow if the environment is not 'gcs'
        print("ENVIRONMENT is 'local' or not set, running local promotion flow. 💻")
        promote_best_flow_local(
            experiment=os.getenv(
                "MLFLOW_EXPERIMENT"
            ),  # You can override the experiment name here if needed
            metric="val_macro_avg_f1-score",  # Override metric if needed
            higher_is_better=True,
            model_name=os.getenv(
                "MODEL_NAME", "passcompass_model"
            ),  # Override model name if needed
            mlflow_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"
            ),  # Pass the URI if it's dynamic
        )


if __name__ == "__main__":
    monthly_flow()
