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

    schema = []
    for raw_col in df.columns.drop("target"):
        if df[raw_col].dtype.kind in "biufc":           # numeric
            schema.append({
                "name": raw_col,
                "kind": "numeric",
                "min":  float(df[raw_col].min()),
                "max":  float(df[raw_col].max()),
            })
        else:                                           # categorical
            schema.append({
                "name":    raw_col,
                "kind":    "categorical",
                "choices": sorted(df[raw_col].dropna().unique().tolist()),
            })


    return X_train, X_val, y_train, y_val, dv, schema

@task
def latest_dataset(base_dir: str = "data/passcompass") -> str:
    """
    Returns the path to the most recent parquet file inside
    data/passcompass/YYYY-MM-DD/.
    """
    root = Path(base_dir)
    # folders sorted by date, newest last
    dated_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not dated_dirs:
        raise FileNotFoundError(f"No YYYY-MM-DD folders in {root!s}")
    newest = dated_dirs[-1]

    # assume exactly one parquet per date; tweak if needed
    parquet_files = list(newest.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet in {newest!s}")

    return str(parquet_files[0])