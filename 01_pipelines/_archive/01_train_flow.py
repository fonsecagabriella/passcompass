from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from model_training import train_and_log_model
from utils import load_data, prepare_features

@task
def get_train_data(path):
    return load_data(path)

@task
def vectorize(df, dv=None):
    return prepare_features(df, dv)

@task
def train_models(X, y, dv):
    for model_type in ["logreg", "gb", "rf"]:
        train_and_log_model(X, y, dv, model_type=model_type)

@flow
def model_training_flow(train_path: str = "data/train.parquet"):
    df = get_train_data(train_path)
    X, y, dv = vectorize(df)
    train_models(X, y, dv)

if __name__ == "__main__":
    model_training_flow()
