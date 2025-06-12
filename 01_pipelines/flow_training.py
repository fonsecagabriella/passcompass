from prefect import task, flow, get_run_logger
import pandas as pd, pathlib, mlflow, os, json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from passcompass_utils.metrics import log_classification_report
from pathlib import Path


# PREFECT SETTINGS
import os
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"
# optional – silence the telemetry SSL warning
os.environ["PREFECT_SEND_ANONYMOUS_TELEMETRY"] = "0"

BASE_DIR = Path(__file__).resolve().parents[1]      # project root
CSV_PATH = BASE_DIR / "data" / "students" / "students_train.csv"

@task
def download_data() -> Path:
    """Return path to fresh CSV (already downloaded in repo)."""
    return CSV_PATH

@task
def preprocess(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    y  = df.pop("pass")
    X  = df.to_dict(orient="records")
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

@task
def train_model(split):
    X_tr, X_val, y_tr, y_val = split
    pipe = Pipeline([
        ("vec", DictVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])
    pipe.fit(X_tr, y_tr)
    return pipe, X_val, y_val

@task
def evaluate(pipe_tuple):
    pipe, X_val, y_val = pipe_tuple
    y_pred = pipe.predict(X_val)
    # log metrics
    log = log_classification_report(y_val, y_pred, prefix="val_")
    return log["val_accuracy"], pipe        # pick key metric to bubble up

@task
def register(pipe, metric):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run():
        mlflow.set_tag("model_type", "LogReg")
        mlflow.log_metric("val_accuracy", metric)
        mlflow.sklearn.log_model(pipe, "model")
    # optional: call your register_best.py or MLflow client here

@flow(name="train_student_model")
def main_flow():
    csv_path = download_data()
    split    = preprocess(csv_path)
    metric, pipe = evaluate(train_model(split))
    register(pipe, metric)

if __name__ == "__main__":
    main_flow()
