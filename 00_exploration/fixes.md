- prefect flow: use tags for comparing models
    - now tags with a dictionary for value doesn't work
- prefect flow: implement triggers for when new data, run new training
- prefect flow: implement tests of other models
- prefect / mlflow / webapp: use pipeline with DictVect and model instead of exporting them separetly
- Webapp - JS - Fix alert for missing fields
- Metrics for success: Review if accuracy is best approach. probably better recall of fail, or implement in flow only to record if above a threshold. Run tests later if reducing decision boundary is a better approach. work with ensemble models to improve performance of general prediciton and fail class.

- Evidently: can't monitor with only one record?
- Evidently: file create_baseline
- Evidently: server
- Evidently: monitor after new data is uploaded
- Evidently: Version control – baseline JSON locks column types; regenerate if your feature list changes.
- Evidently: alert if drift > threshold

- All: rename your pass flag to avoid clash with Python
keyword (students["target"] = students["pass"]


- Mlflow: add calibrated wrapper so all models have predict probability.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
import mlflow.sklearn

base_clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# `cv="prefit"` lets you train once, then calibrate
base_clf.fit(X_train, y_train)

calib_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
calib_clf.fit(X_val, y_val)          # needs some hold-out data

pipeline = make_pipeline(dv, calib_clf)   # include DictVectorizer

mlflow.sklearn.log_model(pipeline, "model", registered_model_name="passcompass_generic")
```

then in the app.py
```python
import numpy as np

def get_pass_probability(model, X):
    """
    Returns a float in [0,1] for a single-row input X.
    Picks the best available method at runtime.
    """
    # A. true probabilities
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])

    # B. decision_function → logistic squashing
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        return float(1 / (1 + np.exp(-score)))   # sigmoid

    # C. last-resort: label → 0 ·25 / 0 ·75 so UI bar still moves
    label = int(model.predict(X)[0])
    return 0.75 if label == 1 else 0.25
```

and in /predict route:

```py
prob_pass = get_pass_probability(model, X_predict)
label     = "Likely to pass" if prob_pass >= 0.5 else "Likely to fail"

return jsonify(probability=round(prob_pass, 3), label=label)
```
