import mlflow
from mlflow.models import Model

mlflow.set_tracking_uri("http://127.0.0.1:5001")


logged_model = 'runs:/ab8b0be1b1b84e12b33782a91057236e/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Load model
model = mlflow.pyfunc.load_model(logged_model)

# Access DictVectorizer
dv = model._model_impl.dv
print(dv.feature_names_)

# Load metrics.json
client = MlflowClient()
metrics_path = client.download_artifacts(run_id, "model/metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)

print(metrics)

