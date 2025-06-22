from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc, os, json
from mlflow.tracking import MlflowClient
from pathlib import Path

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME          = os.getenv("MODEL_NAME", "passcompass_generic")
#MODEL_STAGE         = os.getenv("MODEL_STAGE", "Staging")   # or "Production"
MODEL_STAGE         = "Staging"

app = Flask(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("Loading model…")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "best_202506")

# 1️⃣  Point the client at the same server you open in the browser
mlflow.set_tracking_uri("http://127.0.0.1:5001")      # or http://localhost:5001

# 2️⃣  Now create the client and fetch by alias
client = MlflowClient()

MODEL_NAME  = "passcompass_generic"
MODEL_ALIAS = "best_202506"          # no leading '@'

mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)  # ModelVersion object
run_id = mv.run_id

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

#print(model.metadata)  # print model metadata

print(f"Loading model {MODEL_NAME} v{mv.version} from run {run_id}…")

# ── fetch and cache the feature schema ─────────────────────────────────

#run_id     = model.run_id
local_path = client.download_artifacts(run_id, "feature_schema.json")
with open(local_path) as f:
    FEATURE_SCHEMA = json.load(f)

@app.route("/", methods=["GET"])
def home(): # or def index()
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like:
    {
      "school":"GP","sex":"F","age":17,"studytime":2, ...
    }
    """
    data = request.get_json(force=True)
    prediction = model.predict([data])[0]            # 0 = Fail / 1 = Pass
    #proba      = model.predict_proba([data])[0][1]   # prob of Pass (label 1)

    return jsonify({
        "prediction": int(prediction),
        #"proba_pass": round(float(proba), 3),
        "proba_pass": round(0.78, 3),
        "label": "Pass" if prediction else "Fail"
    })


@app.route("/features", methods=["GET"])
def features():
    schema = [
        {"name": "age", "kind": "numeric", "min": 15, "max": 22},
        {"name": "sex", "kind": "categorical",
         "choices": ["F", "M"]},
        {"name": "address", "kind": "categorical",
         "choices": ["U", "R"]},
        # …
        {"name": "traveltime", "kind": "categorical",
         "choices": [1, 2, 3, 4]},
        {"name": "studytime",  "kind": "categorical",
         "choices": [1, 2, 3, 4]},
        {"name": "failures", "kind": "numeric", "min": 0, "max": 3},
        {"name": "absences", "kind": "numeric", "min": 0, "max": 93},
    ]
    return jsonify(schema)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
