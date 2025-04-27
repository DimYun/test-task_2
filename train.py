"""Module for initiate model training"""
import argparse
import os

import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from configs.config import Config
from src.callbacks.debug import VisualizePredMask
from src.constants import EXPERIMENTS_PATH
from src.datamodule import FirstBreaksDM
from src.lightning_module import FirstBreaksModule

torch.set_float32_matmul_precision('medium')   # 'hight'


def arg_parse() -> argparse.Namespace:
    """
    Parse command line to extract config file path
    :return: dictionary structure with config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file")
    return parser.parse_args()


def train(config: Config) -> None:
    """
    Train the model
    :param config: python module with config values
    :return:
    """
    datamodule = FirstBreaksDM(config)
    model = FirstBreaksModule(config)
    experiment_snapshots_path = EXPERIMENTS_PATH / config.experiment_name
    os.makedirs(experiment_snapshots_path, exist_ok=True)
    task = Task.init(
        project_name=config.project_name,
        task_name=f"{config.experiment_name}",
        auto_connect_frameworks=True,
        output_uri=str(experiment_snapshots_path),
    )
    task.connect(config.dict())
    checkpoint_callback = ModelCheckpoint(
        experiment_snapshots_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=2,
        filename=f"epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}",
    )
    trainer = pl.Trainer(
        max_epochs=config.train_config.n_epochs,
        accelerator=config.train_config.accelerator,
        devices=[config.train_config.device],
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            VisualizePredMask(
                every_n_epochs=3,
                batch_num=config.data_config.batch_size,
                image_width=config.data_config.width,
                image_height=config.data_config.height
            ),
        ],
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == "__main__":
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    train_config = Config.from_yaml(args.config_file)
    train(train_config)
