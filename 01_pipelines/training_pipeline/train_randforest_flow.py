# 01_pipelines/training_pipeline/train_rf_flow.py
from prefect import flow
from hyperopt import hp
from hyperopt.pyll.base import scope

from sklearn.ensemble import RandomForestClassifier

from data_tasks import latest_dataset, load_data, vectorize
from train_utils import run_hpo



ACC_MIN   = 0.78        # or import from constants
MAX_EVALS = 30

@flow(name="train_randforest_flow")
def train_randforest_flow(
    base_data_dir: str = "data/passcompass",
    acc_min: float = ACC_MIN,
):
    # ── 1. Load & vectorise ────────────────────────────────────────────────
    data_path = latest_dataset(base_data_dir)
    df = load_data(data_path)
    X_train, X_val, y_train, y_val, dv, schema = vectorize(df)

    # ── 2. Hyperopt search-space (cast ints in objective – see below) ─────
    search_space = {
        "n_estimators":     scope.int(hp.quniform("n_estimators", 100, 800, 50)),
        "max_depth":        hp.choice("max_depth", [None] + list(range(5, 21, 3))),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
        "min_samples_leaf":  scope.int(hp.quniform("min_samples_leaf", 1, 5, 1)),
        "max_features": hp.choice(
            "max_features", [None, "sqrt", "log2", 0.3, 0.5, 0.8]
        ),
        "class_weight":     hp.choice("class_weight", [None, "balanced"]),
        "bootstrap":        hp.choice("bootstrap", [True, False]),
    }

    # ── 3. Run HPO ─────────────────────────────────────────────────────────
    best = run_hpo(
        RandomForestClassifier,     # <- estimator class
        search_space,
        X_train, y_train, X_val, y_val,
        dv,
        experiment_name="passcompass",
        tag_name="rf",              # every run gets tag model=rf
        acc_min=acc_min,
        max_evals=MAX_EVALS,
        schema=schema
    )
    print("✔️  Random-Forest best params:", best)


if __name__ == "__main__":
    train_randforest_flow()
