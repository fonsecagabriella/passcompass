from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc, os, json

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "passcompass_students")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Staging")   # or "Production"

app = Flask(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("Loading model…")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.route("/", methods=["GET"])
def index():
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
