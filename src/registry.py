from pathlib import Path
import shutil
import mlflow
import pandas as pd

def select_best_run(experiment_name: str, primary_metric: str = "f1") -> dict:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Expérience MLflow '{experiment_name}' introuvable")

    runs: pd.DataFrame = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        raise RuntimeError("Aucun run trouvé.")

    # tri décroissant sur la métrique primaire (plus grand = meilleur)
    metric_col = f"metrics.{primary_metric}"
    runs = runs.sort_values(by=[metric_col], ascending=False, na_position="last")

    best = runs.iloc[0]
    model_name = best.get("tags.mlflow.runName", "unknown")  # Series.get -> lit la colonne
    run_id = best["run_id"]
    score = float(best[metric_col])

    return {"run_id": run_id, "model_name": model_name, "metrics": {primary_metric: score}}

def publish_best_local(best_dir: str, candidates_dir: str, model_name: str):
    src = Path(candidates_dir) / model_name / "model.joblib"
    if not src.exists():
        raise FileNotFoundError(f"Artefact introuvable: {src}")
    dst_dir = Path(best_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.joblib"
    shutil.copy2(src, dst)
    return str(dst)
