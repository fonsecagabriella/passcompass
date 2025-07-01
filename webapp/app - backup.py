from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc, os, json
import pickle
from mlflow.tracking import MlflowClient
from pathlib import Path
import pandas as pd

# ── Flask app setup ────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_STAGE         = "Staging"
MODEL_NAME  = "passcompass_generic"
MODEL_ALIAS = "best_202506"          # no leading '@'

app = Flask(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



# ── fetch and cache the feature schema  ─────────────────────────────────
# Create the client and fetch by alias
client = MlflowClient()



mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)  # ModelVersion object
run_id = mv.run_id

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

#print(model.metadata)  # print model metadata

print(f"Loading model {MODEL_NAME} v{mv.version} from run {run_id}…")


local_path = client.download_artifacts(run_id, "feature_schema.json")
with open(local_path) as f:
    FEATURE_SCHEMA = json.load(f)

NUMERIC_COLS = {col["name"] for col in FEATURE_SCHEMA if col["kind"] == "numeric"}


# --- fetch DictVectorizer artifact -----------------------
dv_dir = client.download_artifacts(mv.run_id, "dv.pkl")
dv_path = os.path.join(dv_dir, "dv.pkl")  
with open(dv_path, "rb") as f:
    dv = pickle.load(f)


# ── define app routes ─────────────────────────────────

@app.route("/", methods=["GET"])
def home(): # or def index()
    return render_template("index.html")

@app.post("/predict")
def predict():
    raw = request.get_json()            # dict from JS
    if raw is None:
        return jsonify(error="No JSON payload"), 400

    # ---- 1. cast numeric strings → float ----------------------
    casted = {
        k: (float(v) if k in NUMERIC_COLS else v)
        for k, v in raw.items()
    }

    # ---- 2. wrap in DataFrame --------------------------------
    df = pd.DataFrame([casted])

    print("Received data:", df)

    # ---- 2.1. apply dict vectorizer ----------------------
    #X_predict = dv.transform(df)
    X_predict = dv.transform([casted]).toarray()

    # ---- 2.5. check for missing values (future implementation) -----------------------

    # ---- 3. predict ------------------------------------------
    print(X_predict.shape, X_predict.dtype )
    proba_pass = float(model.predict(X_predict))
    #proba_pass = float(model.predict_proba(df)[0, 1])    # col 1 = pass
    label      = "Likely to pass" if proba_pass >= 0.5 else "Likely to fail"

    # ---- 4. respond ------------------------------------------
    return jsonify(probability=round(proba_pass, 3), label=label)


@app.route("/features", methods=["GET"])
def get_features():
    print(FEATURE_SCHEMA)
    try:
        return jsonify(FEATURE_SCHEMA)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
