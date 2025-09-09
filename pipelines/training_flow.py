from prefect import flow, task, get_run_logger
from omegaconf import DictConfig
from hydra import compose, initialize
from src.utils import setup_mlflow
from src.train import train_one_model
from src.registry import select_best_run, publish_best_local
from omegaconf import OmegaConf

@task
def setup(cfg: DictConfig):
    setup_mlflow(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)

@task
def train_model_task(cfg, model_cfg):
    return train_one_model(cfg, model_cfg)

@task
def select_and_publish_task(cfg):
    best = select_best_run(cfg.mlflow.experiment_name, cfg.metrics.primary)
    path = publish_best_local(cfg.paths.best_dir, cfg.paths.models_dir, best["model_name"])
    return {"best": best, "path": path}

@flow(name="lab1-multi-model-training")
def training_flow(cfg: DictConfig):
    logger = get_run_logger()
    setup(cfg)
    results = []
    for model_name in cfg.run.models:
        model_cfg = OmegaConf.load(f"configs/models/{model_name}.yaml")
        r = train_model_task.submit(cfg, model_cfg)
        results.append(r)
    results = [r.result() for r in results]
    for r in results:
        logger.info(f"{r['name']} -> {r['metrics']}")
    best = select_and_publish_task(cfg)
    logger.info(f"Best -> {best}")

def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="base")
    training_flow(cfg)

if __name__ == "__main__":
    main()
