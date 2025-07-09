from hyperopt import hp
from prefect import flow
from sklearn.linear_model import LogisticRegression

from .data_tasks import _resolve_data_uri, latest_dataset, load_data, vectorize
from .train_utils import run_hpo

# ─── you will overwrite this from Prefect CLI or env var ──────────────
ACC_MIN = 0.78  #  ←  set later!
MAX_EVALS = 25
# ----------------------------------------------------------------------


@flow(name="train_logreg_flow")
def train_logreg_flow(
    base_data_dir: str = "data/passcompass",
    acc_min: float = ACC_MIN,
):

    data_path = latest_dataset(base_data_dir)
    df = load_data(data_path)
    X_train, X_val, y_train, y_val, dv, schema = vectorize(df)

    search_space = {
        "C": hp.loguniform("C", -7, 4),  #  e^(−7)…e^(4)
        "penalty": hp.choice("penalty", ["l1", "l2"]),
        "class_weight": hp.choice("cw", [None, "balanced"]),
        "solver": "liblinear",
        "max_iter": 500,
    }

    best = run_hpo(
        LogisticRegression,
        search_space,
        X_train,
        y_train,
        X_val,
        y_val,
        dv,
        experiment_name="passcompass_mlops",
        tag_name="logreg",
        # tags={"model": "logreg"},
        acc_min=acc_min,
        max_evals=MAX_EVALS,
        schema=schema,
    )
    print("✔️  Best params:", best)


if __name__ == "__main__":
    data_uri = _resolve_data_uri()
    train_logreg_flow(base_data_dir=data_uri)
