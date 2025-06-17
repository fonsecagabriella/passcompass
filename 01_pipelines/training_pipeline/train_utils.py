import json
import mlflow
import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve

from passcompass_utils.metrics import (
    log_classification_report,
    evaluate_and_log,
)

def _best_threshold(y_true, prob_fail, acc_min):
    """
    Sweep thresholds; pick the one with highest recall_fail
    while maintaining accuracy >= acc_min.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr, best_rec, best_acc = 0.5, 0.0, 0.0

    for thr in thresholds:
        # predict label 0 ("fail") if prob_fail >= thr, else 1 ("pass")
        y_pred = (prob_fail >= thr).astype(int)          # 1 if True else 0
        y_pred = 1 - y_pred                              # invert → 0=fail,1=pass
        acc  = accuracy_score(y_true, y_pred)
        rec0 = recall_score(y_true, y_pred, pos_label=0)

        if acc >= acc_min and rec0 > best_rec:
            best_thr, best_rec, best_acc = thr, rec0, acc

    return best_thr, best_rec, best_acc


def run_hpo(
    model_cls,
    search_space,
    X_train,
    y_train,
    X_val,
    y_val,
    dv,
    experiment_name: str,
    tag_name: str,
    acc_min: float,                   # <-- external variable
    max_evals: int = 30,
    random_state=None,
):
    """
    One Hyperopt loop that   (i) tunes hyper-parameters,
    (ii) tunes a decision threshold *after* training,
    (iii) logs only models whose tuned accuracy >= acc_min.
    """

    mlflow.set_experiment(experiment_name)

    def objective(params):
        with mlflow.start_run(nested=True, tags={"model": tag_name}):
            # --------  train
            model = model_cls(**params)
            model.fit(X_train, y_train)

            # --------  probability of *fail* (label 0)
            idx_fail = list(model.classes_).index(0)
            prob_fail = model.predict_proba(X_val)[:, idx_fail]

            # --------  threshold sweep
            thr, rec0, acc = _best_threshold(y_val, prob_fail, acc_min)

            # --------  log metrics
            mlflow.log_param("threshold", thr)
            mlflow.log_metrics({
                "val_recall_fail_tuned": rec0,
                "val_accuracy_tuned":    acc,
            })

            # full report (uses tuned threshold)
            if prob_fail >= thr:
                y_pred_tuned = 1
            else:
                y_pred_tuned = 0
            
            #y_pred_tuned = (prob_fail >= thr).astype(int == 0)

            log_classification_report(
                y_val, y_pred_tuned, prefix="val_"
            )

            mlflow.log_params(params)
            mlflow.log_param(
                "num_features", len(dv.feature_names_)
            )
            mlflow.set_tag("feature_list",
                           json.dumps(dv.feature_names_.tolist()))

            # --------  optionally save the model
            if acc >= acc_min:
                mlflow.sklearn.log_model(
                    model, "model",
                    input_example=X_train[:1],
                    registered_model_name=None,
                    extra_pip_requirements=["scikit-learn"]
                )

            # Hyperopt tries to *minimise* → negative recall of fail class
            return {"loss": -rec0, "status": STATUS_OK}

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
    )
    return best_params
