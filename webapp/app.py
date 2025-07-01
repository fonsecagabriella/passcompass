# app.py  ─────────────────────────────────────────────────────────────
"""
PassCompass predictor – Flask version
-------------------------------------

Quick start (local model):
    export MODEL_SOURCE=local
    export LOCAL_MODEL_PATH=./model            # folder with MLmodel
    python app.py
    # open http://127.0.0.1:8080

Use a model stored in GCS:
    export MODEL_SOURCE=gcs
    export GCS_MODEL_URI=gs://passcompass-ml-bucket/model/passcompass_generic_v12
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json   # if not on GCP
    python app.py
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any, Dict, List

import mlflow
#import mlflow.pyfunc
import pickle
from mlflow.tracking import MlflowClient

from google.cloud import storage                # NEW


from dotenv import load_dotenv        # make sure you `pip install python-dotenv`
load_dotenv()                         # picks up .env automatically

import numpy as np
import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# ─────────────────────────── Configuration ──────────────────────────
# Resolve data location based on ENVIRONMENT
#ENVIRONMENT   = os.getenv("ENVIRONMENT", "local").lower()
ENVIRONMENT   = "local"
# for local development, model lives in MLflow
MODEL_NAME  = "passcompass_generic"
MODEL_ALIAS = "best_202506"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
LOCAL_MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}" 
dv = None                                        # global singleton
# for retrieving model from GCS
GCS_MODEL_URI   = os.getenv("GCS_MODEL_URI", "gs://passcompass-ml-bucket/model/passcompass_generic_v12")

if ENVIRONMENT == "local" and not LOCAL_MODEL_URI:
    raise RuntimeError("ENVIRONMENT=local but LOCAL_MODEL_URI is not set")
if ENVIRONMENT == "gcs" and not GCS_MODEL_URI:
    raise RuntimeError("ENVIRONMENT=gcs but GCS_MODEL_URI is not set")



# ─────────────────────────── Flask setup ────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

# Global (per worker) singletons
model: mlflow.pyfunc.PyFuncModel | None = None
schema: List[Dict[str, Any]] = []            # list of {"name":str,"kind":str,…}


# ─────────────────────── Helpers & bootstrap ────────────────────────
def _resolve_model_uri() -> str:
    if ENVIRONMENT == "local":
        print(f"Using local data URI: {LOCAL_MODEL_URI}")
        app.logger.info(f"Loading model from MLflow Registry: {LOCAL_MODEL_URI}")
        return LOCAL_MODEL_URI
    elif ENVIRONMENT == "gcs":
        print(f"Using GCS model URI: {GCS_MODEL_URI}")
        return GCS_MODEL_URI
    else:
        raise ValueError(f"Unsupported ENVIRONMENT '{ENVIRONMENT}'")

def _download_dv(uri: str) -> str:
    """
    Downloads dv.pkl from either layout:
        gs://bucket/.../model/dv.pkl
        gs://bucket/.../model/dv.pkl/dv.pkl
    and returns the local file path.
    Works for Registry URIs too.
    """
    # 1️⃣  Normalise parent folder that holds MLmodel
    parent = uri[:-6] if uri.rstrip("/").endswith("/model") else uri

    # 2️⃣  Registry path → let mlflow handle; single layout is enough
    if parent.startswith("models:/"):
        try:
            return mlflow.artifacts.download_artifacts(f"{parent}/dv.pkl")
        except Exception:
            return mlflow.artifacts.download_artifacts(f"{parent}/dv.pkl/dv.pkl")

    # 3️⃣  GCS path(s)
    if parent.startswith("gs://"):
        bucket, key = parent[5:].split("/", 1)
        client  = storage.Client()
        bucket  = client.bucket(bucket)

        for blob_key in (f"{key.rstrip('/')}/dv.pkl",
                         f"{key.rstrip('/')}/dv.pkl/dv.pkl"):
            blob = bucket.blob(blob_key)
            if blob.exists():
                tmp_dir = tempfile.mkdtemp()
                local   = os.path.join(tmp_dir, "dv.pkl")
                # make sure parent dirs exist
                os.makedirs(os.path.dirname(local), exist_ok=True)
                blob.download_to_filename(local)
                return local

        raise FileNotFoundError(f"dv.pkl not found under {uri}")

    # 4️⃣  Local filesystem path
    path1 = pathlib.Path(parent) / "dv.pkl"
    path2 = path1 / "dv.pkl"
    for p in (path1, path2):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"dv.pkl not found under {uri}")


# def _local_model_uri():
#     """
#     Returns the local model URI based on the environment.
#     This is used for local development/testing.
#     """
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#     # ── fetch and cache the feature schema  ─────────────────────────────────
#     # Create the client and fetch by alias
#     client = MlflowClient()

#     mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)  # ModelVersion object
#     run_id = mv.run_id

#     model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

#     print(model.metadata)  # print model metadata

#     print(f"Loading model {MODEL_NAME} v{mv.version} from run {run_id}…")
#     return f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

def _load_model_and_schema() -> tuple[mlflow.pyfunc.PyFuncModel, list]:
    """
    Loads the MLflow model (Registry or GCS) and returns (model, schema).

    Search order for schema:
      1. feature_schema.json file inside the model artifacts
      2. features.json  (legacy name)
      3. Fallback: derive a minimal schema from MLflow signature
    """
    # 1️⃣ Decide URI
    # if ENVIRONMENT == "local":
    #     uri = LOCAL_MODEL_URI
    #     app.logger.info(f"Loading model from MLflow Registry: {uri}")
    # else:  # gcs
    #     uri = GCS_MODEL_URI
    #     app.logger.info(f"Loading model from GCS: {uri}")

    uri = _resolve_model_uri()
    app.logger.info(f"Resolved model URI: {uri}")
 
    global dv
    if dv is None:
        dv_path = _download_dv(uri)
        if os.path.isdir(dv_path):
            candidate = os.path.join(dv_path, "dv.pkl")
            if os.path.exists(candidate):
                dv_path = candidate
            else:
                raise FileNotFoundError(f"dv.pkl not found inside {dv_path}")

        print(f"DictVectorizer downloaded to {dv_path}")
        with open(dv_path, "rb") as f:
            dv = pickle.load(f)
        app.logger.info("DictVectorizer loaded")

    # 2️⃣ Download *just* the schema file if it exists
    schema_path: str | None = None
    for name in ("feature_schema.json", "features.json"):
        try:
            schema_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{uri}/{name}"
            )
            if os.path.exists(schema_path):
                break
        except Exception:
            schema_path = None

    # 3️⃣ Load model itself
    model = mlflow.pyfunc.load_model(uri)


    # 4️⃣ If we found a schema file, use it
    if schema_path and os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
            print(schema)  # print model metadata
        return model, schema

    # 5️⃣ Fallback: derive schema from signature
    sig = model.metadata.get_input_schema()
    schema = []
    for col in sig.inputs:
        dtype = str(col.type)
        kind = "numeric" if dtype.startswith(("int", "float", "double")) else "categorical"
        schema.append({"name": col.name, "kind": kind})

    return model, schema


def _startup_once() -> None:
    """Load model + schema exactly once per worker process."""
    global model, schema
    model, schema = _load_model_and_schema()
    app.logger.info(
        f"Model loaded from {ENVIRONMENT.upper()}  |  "
        f"{len(schema)} features"
    )


# Eager load at module import (works for Flask dev server, gunicorn, Cloud Run)
_startup_once()

# ────────────────────────── Routes ───────────────────────────────────
@app.route("/")
def index():
    """HTML front-end."""
    return render_template("index.html")


@app.route("/features")
def features():
    """Expose feature schema for the JS front-end."""
    return jsonify(schema)



@app.post("/predict")
def predict():
    """Score a single sample.  Expects JSON with feature names as keys."""
    if not request.is_json:
        abort(400, "Payload must be JSON")

    raw = request.get_json()
    casted = {
        k: (float(v) if any(f["name"]==k and f["kind"]=="numeric" for f in schema) else v)
        for k, v in raw.items()
    }

    # vectorise with the DictVectorizer we loaded above
    X_vec = dv.transform([casted]).toarray()

    proba_pass = float(model.predict(X_vec))          # your model outputs P(pass)
    return jsonify(probability=round(proba_pass, 6))



@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve JS/CSS/assets residing in ./static/ ."""
    return send_from_directory(app.static_folder, filename)


# ───────────────────────── Entry-point ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
