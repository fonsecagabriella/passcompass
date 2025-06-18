from prefect import flow
from hyperopt import hp
from hyperopt.pyll.base import scope

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

from data_tasks import latest_dataset, load_data, vectorize
from train_utils import run_hpo



ACC_MIN   = 0.78        # or import from constants
MAX_EVALS = 30

@flow(name="train_gbd_flow")
def train_gbd_flow(
    base_data_dir: str = "data/passcompass",
    acc_min: float = ACC_MIN,
):
    # ── 1. Load & vectorise ────────────────────────────────────────────────
    data_path = latest_dataset(base_data_dir)
    df = load_data(data_path)
    X_train, X_val, y_train, y_val, dv = vectorize(df)

    # ── 2. Hyperopt search-space (cast ints in objective – see below) ─────
    search_space = {
        # int
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 50)),
        "max_depth":    hp.choice("max_depth", [None] + list(range(2, 7))),  # ints already
        # floats OK
        "learning_rate": hp.loguniform("learning_rate", -4, -0.7),
        "subsample":     hp.uniform("subsample", 0.6, 1.0),
        "max_features":  hp.choice("max_features",
                                [None, "sqrt", "log2", 0.3, 0.5, 0.8]),
    }

    # ── 3. Run HPO ─────────────────────────────────────────────────────────
    best = run_hpo(
        GradientBoostingClassifier,     # <- estimator class
        search_space,
        X_train, y_train, X_val, y_val,
        dv,
        experiment_name="passcompass",
        tag_name="gbc",              # every run gets tag model=rf
        acc_min=acc_min,
        max_evals=MAX_EVALS,
    )
    print("✔️  GBC best params:", best)


if __name__ == "__main__":
    train_gbd_flow()
