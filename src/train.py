from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from .data import load_data, split_xy, train_test_split_xy
from .preprocess import build_preprocessor
from .evaluate import compute_metrics
from .utils import import_estimator, save_artifacts_local

def train_one_model(cfg, model_cfg: dict):
    df = load_data(cfg.paths.raw)
    X, y = split_xy(df, cfg.data.target)
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    pre = build_preprocessor(num_cols, cat_cols, use_feature_eng=cfg.features.get("feature_engineering", True))

    Est = import_estimator(model_cfg["estimator"])
    clf = Est(**(model_cfg.get("params") or {}))
    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

    with mlflow.start_run(run_name=model_cfg["name"]) as run:
        mlflow.log_params({
            "model": model_cfg["name"],
            **{f"clf__{k}": v for k, v in (model_cfg.get("params") or {}).items()},
            "feature_engineering": cfg.features.get("feature_engineering", True),
        })
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe.named_steps["clf"], "predict_proba") else None

        m = compute_metrics(y_test, y_pred, y_proba, pos_label=cfg.metrics.pos_label)
        mlflow.log_metrics(m)

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        out_dir = Path(cfg.paths.models_dir) / model_cfg["name"]
        save_artifacts_local(pipe, str(out_dir), extras={"metrics": m, "run_id": run.info.run_id})

        return {"run_id": run.info.run_id, "name": model_cfg["name"], "metrics": m}
