# 🚀 Summary of Action Items (not necessarily in priority order)

🚨 The project current only **exports the model & evidently reports** to the cloud. (2025, 07)

- `prefect flow`: Add a monitoring trigger or alert simulation
- `prefect flow`: Deploy or register orchestration (Prefect, masterflow)
- `cloud`: upload data to cloud and read data from cloud function (training)
- `prefect flow`: unify flows `promote best cloud` and `promote best local`
- `prefect flow`: use tags for comparing models
    - now tags with a dictionary for value doesn't work
- `prefect flow`: implement triggers for when new data, run new training
- `prefect flow`: implement tests of other models
- `prefect` / `mlflow` / `webapp`: use pipeline with DictVect and model instead of exporting them separetly
- `webapp` - JS - Fix alert for missing fields
- `mlflow` `experimentation` Metrics for success: Review if accuracy is best approach. probably better recall of fail, or implement in flow only to record if above a threshold. Run tests later if reducing decision boundary is a better approach. work with ensemble models to improve performance of general prediciton and fail class.
- `evidently`: server
- `evidently`: monitor after new data is uploaded
- `evidently`: Version control – baseline JSON locks column types; regenerate if your feature list changes.
- `evidently`: alert if drift > threshold
- `evidently`: good tutorial: https://medium.com/@elmahfoudradwane/a-comprehensive-guide-to-mlflow-for-data-scientists-real-world-applications-and-examples-d4a2a32dda22
- `data`: rename your pass flag to avoid clash with Python
    - keyword (students["target"] = students["pass"]
- `mlflow`: add calibrated wrapper so all models have predict probability.
- `mlflow`: Pre-seed MLflow with a “good” model (say val_macro_avg_f1=0.8, alias=best). Then run a flow that logs a worse score (0.6). Assert that alias "best" still points to the old version.


## Notes

```bash
MASTERFLOW

Calls the extract flow.

Calls the training and model selection flow.

Runs the Evidently baseline flow (once).

Runs the Evidently monitoring flow (on each run).

Deploys the model.

Optionally schedules everything.
```

To run the masterflow:

- make prefect and mlflow server
- run masterflow
- copy docker image and run webserver to predict



