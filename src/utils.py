import importlib
from pathlib import Path
import mlflow
import joblib
import json

def import_estimator(path: str):
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)

def setup_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def save_artifacts_local(model, out_dir: str, filename: str = "model.joblib", extras: dict = None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(Path(out_dir)/filename))
    if extras:
        with open(Path(out_dir)/"meta.json", "w") as f:
            json.dump(extras, f, indent=2)
