🚀 Summary of Action Items (in priority order)
✅ Containerize and deploy your model (Cloud Run or Docker locally)

✅ Add requirements.txt, full README.md, dataset link

✅ Add a monitoring trigger or alert simulation

✅ Deploy or register orchestration (Prefect/Airflow)

✅ Add Makefile, CI/CD, pre-commit, linter

✅ Add at least 1 unit and 1 integration test

the project current only **exports the model** to the cloud.




- ensure all files are reading .env and prepare that for when sending to cloud
- cloud: upload data to cloud and read data from cloud function (training)
- unify promote best cloud and promote best local
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
- Evidently: good tutorial: https://medium.com/@elmahfoudradwane/a-comprehensive-guide-to-mlflow-for-data-scientists-real-world-applications-and-examples-d4a2a32dda22

- All: rename your pass flag to avoid clash with Python
keyword (students["target"] = students["pass"]


- Mlflow: add calibrated wrapper so all models have predict probability.
