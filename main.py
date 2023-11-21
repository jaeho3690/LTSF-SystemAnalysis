import os
import sys
import pickle
import datetime
import logging
import matplotlib.pyplot as plt

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, LearningRateMonitor

from utils.tools import print_options
from dataset.datamodule import LTSFDataModule
from lightning_models.lightning_pl import LitModel

torch.set_printoptions(sci_mode=False)
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print_options(cfg)
    checkpoint_path = os.path.join("checkpoints/", str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    log.info(f"Training {cfg.model.model_name} model on {cfg.data.data_name} dataset")
    log.info(f"Key params: seq_len {cfg.seq_len}, label_len {cfg.label_len}, pred_len {cfg.pred_len}")
    dm = LTSFDataModule(cfg)
    dm.setup(stage="fit")
    model = LitModel(cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor=cfg.optimization.callbacks.monitor,
        save_top_k=1,
        filename=cfg.model.model_name + "-{epoch:02d}-{val_loss:.2f}",
        mode=cfg.optimization.callbacks.mode,
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.optimization.callbacks.monitor,
        min_delta=cfg.optimization.callbacks.min_delta,
        patience=cfg.optimization.callbacks.patience,
        verbose=True,
        mode=cfg.optimization.callbacks.mode,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[cfg.gpu_id],
        benchmark=cfg.benchmark,
        check_val_every_n_epoch=cfg.optimization.callbacks.check_val_every_n_epoch,
        max_epochs=cfg.optimization.callbacks.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor_callback],
        fast_dev_run=cfg.fast_dev_run,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    log.info("Start training")
    trainer.fit(model, dm)

    topk_checkpoint_paths = os.listdir(checkpoint_path)
    dm.setup(stage="test")
    trainer.test(model, dm.test_dataloader(), ckpt_path=checkpoint_path + "/" + topk_checkpoint_paths[0])


def build_model(cfg):
    if cfg.model.model_name == "Transformer":
        from models.Transformer import Model

        model = Model(cfg)
    elif cfg.model.model_name == "Autoformer":
        from models.Autoformer import Model

        model = Model(cfg)
    elif cfg.model.model_name == "Informer":
        from models.Informer import Model

        model = Model(cfg)
    elif cfg.model.model_name == "Dlinear":
        from models.Dlinear import Model

        model = Model(cfg)
    else:
        raise NotImplementedError
    return model


if __name__ == "__main__":
    # Set hyrda configuration not to change the directory by default. This is needed for the output directory to work.
    sys.argv.append("hydra.job.chdir=False")
    main()
