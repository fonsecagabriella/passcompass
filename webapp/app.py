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
import pickle
import tempfile
from typing import Any

import mlflow
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# ─────────────────────────── Configuration ──────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "local").lower()

MODEL_NAME = os.getenv("MODEL_NAME", "passcompass_generic")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "best")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

LOCAL_MODEL_URI = os.getenv(
    "LOCAL_MODEL_URI",
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
)

GCS_MODEL_URI = os.getenv("GCS_MODEL_URI")


# ─────────────────────────── Flask setup ────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

# Global (per worker) singletons
model: mlflow.pyfunc.PyFuncModel | None = None
schema: list[dict[str, Any]] = []
dv: pickle.Pickler | None = None


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

    # Get the MLflow tracking URI from environment variables
    # This is crucial for telling the client where to download artifacts from
    current_mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not current_mlflow_tracking_uri:
        # Fallback to default if not set, though it should be set by docker run
        current_mlflow_tracking_uri = "http://127.0.0.1:5001"

    # 2️⃣  Registry path → let mlflow handle; single layout is enough
    if parent.startswith("models:/"):
        # Explicitly set the tracking URI for the MLflow client
        # before attempting to download artifacts. This is done in _startup_once
        # but also explicitly passed here for robustness.
        # mlflow.set_tracking_uri(current_mlflow_tracking_uri) # This line is handled by _startup_once

        try:
            # When downloading from 'models:/' URI, MLflow client will use the
            # currently set tracking URI to communicate with the server to get artifacts.
            # Explicitly pass tracking_uri to force HTTP download from the server.
            return mlflow.artifacts.download_artifacts(
                f"{parent}/dv.pkl", tracking_uri=current_mlflow_tracking_uri
            )
        except Exception:
            # Explicitly pass tracking_uri to force HTTP download from the server.
            return mlflow.artifacts.download_artifacts(
                f"{parent}/dv.pkl/dv.pkl", tracking_uri=current_mlflow_tracking_uri
            )

    # 3️⃣  GCS path(s)
    if parent.startswith("gs://"):
        bucket, key = parent[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket)

        for blob_key in (f"{key.rstrip('/')}/dv.pkl", f"{key.rstrip('/')}/dv.pkl/dv.pkl"):
            blob = bucket.blob(blob_key)
            if blob.exists():
                tmp_dir = tempfile.mkdtemp()
                local = os.path.join(tmp_dir, "dv.pkl")
                # make sure parent dirs exist
                os.makedirs(os.path.dirname(local), exist_ok=True)
                blob.download_to_filename(local)
                return local

        raise FileNotFoundError(f"dv.pkl not found under {uri}")

    # 4️⃣  Local filesystem path
    # This block is problematic for Docker containers if the path is on the host.
    # It's generally better to rely on MLflow server for artifacts in a containerized app.
    path1 = pathlib.Path(parent) / "dv.pkl"
    path2 = path1 / "dv.pkl"
    for p in (path1, path2):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"dv.pkl not found under {uri}")


def _load_model_and_schema() -> tuple[mlflow.pyfunc.PyFuncModel, list]:
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
            # Explicitly pass tracking_uri for schema download as well
            schema_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{uri}/{name}",
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),  # Explicitly pass tracking_uri
            )
            if os.path.exists(schema_path):
                break
        except Exception:
            schema_path = None

    # 3️⃣ Load model itself
    # mlflow.pyfunc.load_model also respects mlflow.set_tracking_uri()
    model = mlflow.pyfunc.load_model(uri)

    # 4️⃣ If we found a schema file, use it
    if schema_path and os.path.exists(schema_path):
        with open(schema_path) as f:
            schema = json.load(f)
            print(schema)
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
    # Ensure MLflow tracking URI is set at startup for all MLflow operations
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model, schema = _load_model_and_schema()
    app.logger.info(f"Model loaded from {ENVIRONMENT.upper()}  |  " f"{len(schema)} features")


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
        k: (float(v) if any(f["name"] == k and f["kind"] == "numeric" for f in schema) else v)
        for k, v in raw.items()
    }

    # vectorise with the DictVectorizer we loaded above
    X_vec = dv.transform([casted]).toarray()

    proba_pass = float(model.predict(X_vec))
    return jsonify(probability=round(proba_pass, 6))


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve JS/CSS/assets residing in ./static/ ."""
    return send_from_directory(app.static_folder, filename)


# ───────────────────────── Entry-point ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
