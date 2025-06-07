"""
Reusable helpers for model evaluation & MLflow logging.
"""

from __future__ import annotations
import json, tempfile, pathlib
from typing import Sequence

import mlflow
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def evaluate_and_log(
    model,                     # fitted estimator with predict / predict_proba
    X_test, y_test,
    *,
    run,                       # active mlflow run (mlflow.start_run()) or None
    positive_label: int = 1,
    feature_names: Sequence[str] | None = None,
    prefix: str = ""
    ) -> dict:
    """
    Compute metrics & log them to MLflow.

    Returns a dict of metric_name -> value for convenience.
    """
    probas = getattr(model, "predict_proba")(X_test)[:, positive_label]
    preds  = model.predict(X_test)

    metrics = {
        f"{prefix}roc_auc":      roc_auc_score(y_test, probas),
        f"{prefix}f1_macro":     f1_score(y_test, preds, average="macro"),
        f"{prefix}accuracy":     model.score(X_test, y_test),
        f"{prefix}f1_fail":      f1_score(y_test, preds, pos_label=1 - positive_label),
        f"{prefix}precision_fail": precision_score(y_test, preds, pos_label=1 - positive_label),
        f"{prefix}recall_fail":    recall_score(y_test, preds, pos_label=1 - positive_label),
    }

    if run is None:
        run = mlflow.active_run()

    if run:
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # --- log confusion-matrix heat-map as an artifact ---
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()

        with tempfile.TemporaryDirectory() as tmp:
            img_path = pathlib.Path(tmp) / "confusion_matrix.png"
            fig.savefig(img_path)
            mlflow.log_artifact(str(img_path), artifact_path="plots")

        plt.close(fig)

        # --- log feature list (good for feature-selection experiments) ---
        if feature_names is not None:
            mlflow.log_param("num_features", len(feature_names))
            mlflow.set_tag("feature_list", json.dumps(list(feature_names)))

    return metrics




def _flatten_report(report: Mapping[str, dict], prefix: str = "") -> dict:
    """
    Turn sklearn's nested classification_report dict into
    {f"{prefix}{section}_{metric}": value}.
    """
    flat = {}
    for section, metrics in report.items():
        if isinstance(metrics, dict):
            for m_name, val in metrics.items():
                # e.g. "fail_precision" or "macro_avg_f1-score"
                key = f"{prefix}{section.replace(' ', '_')}_{m_name}"
                flat[key] = val
        else:
            # accuracy is just a float
            flat[f"{prefix}accuracy"] = metrics
    return flat

def log_classification_report(
    y_true,
    y_pred,
    run=None,
    *,
    prefix: str = "",
    artifact_path: str = "reports"
    ) -> dict:
    """
    Compute + log the entire sklearn classification_report to MLflow.

    Returns the flattened metrics dict for immediate use.
    """
    report_dict = classification_report(
        y_true, y_pred,
        target_names=None,      # keep numeric labels
        output_dict=True,
        zero_division=0
    )

    metrics = _flatten_report(report_dict, prefix=prefix)

    # --- log to MLflow ---
    run = run or mlflow.active_run()
    if run:
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # save full JSON artifact
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "classification_report.json"
            path.write_text(json.dumps(report_dict, indent=2))
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    return metrics