.PHONY: train mlflow-ui clean data

train:
\tpython -m pipelines.training_flow

mlflow-ui:
\tmlflow ui --backend-store-uri file:./mlruns --port 5000

data:
\tpython scripts/export_breast_cancer.py

clean:
\trm -rf mlruns models/candidates models/best reports/metrics __pycache__ .pytest_cache
