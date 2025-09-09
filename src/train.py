from pathlib import Path
import io

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from .data import load_data, split_xy, train_test_split_xy
from .preprocess import build_preprocessor
from .evaluate import compute_metrics
from .utils import import_estimator, save_artifacts_local

def train_one_model(cfg, model_cfg: dict):
    # 1) données
    df = load_data(cfg.paths.raw)
    X, y = split_xy(df, cfg.data.target)
    X_train, X_test, y_train, y_test = train_test_split_xy(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    # 2) colonnes (auto)
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # 3) preprocess (+ feature engineering si activé)
    pre = build_preprocessor(num_cols, cat_cols, use_feature_eng=cfg.features.get("feature_engineering", True))

    # 4) modèle
    Est = import_estimator(model_cfg["estimator"])
    clf = Est(**(model_cfg.get("params") or {}))

    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

    # 5) MLflow run
    with mlflow.start_run(run_name=model_cfg["name"]) as run:
        # Log params (y compris hyperparams et flag feature eng)
        mlflow.log_params({
            "model": model_cfg["name"],
            **{f"clf__{k}": v for k, v in (model_cfg.get("params") or {}).items()},
            "feature_engineering": cfg.features.get("feature_engineering", True),
        })

        # ========== EDA demandée dans le TP ==========
        # Distribution de la cible (sur train)
        fig = plt.figure()
        pd.Series(y_train).value_counts().sort_index().plot(kind="bar")
        plt.title("Target distribution (train)")
        plt.xlabel("class")
        plt.ylabel("count")
        plt.tight_layout()
        mlflow.log_figure(fig, "eda_target_distribution.png")
        plt.close(fig)

        # Mutual Information sur features transformées (prep uniquement)
        prep_only = pipe.named_steps["prep"]
        # on fit le preprocess sur train via pipe.fit plus bas; ici on transformera après le fit pour rester cohérent
        # =============================================

        # Fit du pipeline complet
        pipe.fit(X_train, y_train)

        # Maintenant qu'il est fit, on peut transformer X_train avec le preprocess
        X_train_trans = prep_only.transform(X_train)
        try:
            mi = mutual_info_classif(X_train_trans, y_train, random_state=cfg.data.random_state)
            fig = plt.figure(figsize=(6, 4))
            idx = np.argsort(mi)[::-1][:20]
            plt.barh(range(len(idx)), np.array(mi)[idx][::-1])
            plt.yticks(range(len(idx)), [f"f{i}" for i in idx][::-1])
            plt.title("Mutual information (top 20 features)")
            plt.tight_layout()
            mlflow.log_figure(fig, "eda_mutual_info.png")
            plt.close(fig)
        except Exception as e:
            mlflow.log_text(f"MI plotting skipped due to: {e}", "eda_mi_warning.txt")

        # Prédictions
        y_pred = pipe.p_
