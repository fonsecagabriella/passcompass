from prefect import flow
from training_pipeline.train_logreg_flow import train_logreg_flow
from training_pipeline.train_rf_flow import train_randforest_flow
from training_pipeline.train_gbc_flow import train_gbc_flow
from promotion_pipeline.promote_best_flow import promote_best_model_flow

@flow(name="daily_training_plus_promotion")
def daily_flow():
    train_logreg_flow()
    train_randforest_flow()
    train_gbc_flow()
    promote_best_model_flow()
