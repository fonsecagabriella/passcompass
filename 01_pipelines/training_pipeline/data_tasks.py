from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from prefect import task

@task
def load_data(path: str | Path):
    return pd.read_parquet(path)

@task
def vectorize(df, target_col: str = "pass"):
    """
    Returns X_train, X_val, y_train, y_val, DictVectorizer
    """
    y = df[target_col].values
    dicts = df.drop(columns=[target_col]).to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val, dv
