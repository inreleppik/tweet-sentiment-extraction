import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split

from tweet_sentiment_extraction.modules.data_module import TweetDataModule
from tweet_sentiment_extraction.modules.module import TweetLightningModule
from tweet_sentiment_extraction.utils import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    pl.seed_everything(cfg.training.seed, workers=True)

    train_csv_path = to_absolute_path(cfg.data_loading.train_csv)
    df = pd.read_csv(train_csv_path)
    df["text"] = df["text"].astype(str)
    df["selected_text"] = df["selected_text"].astype(str)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=cfg.training.seed,
        stratify=df["sentiment"],
    )

    dm = TweetDataModule(
        train_df=train_df,
        val_df=val_df,
        batch_size=cfg.training.batch_size,
        model_name=cfg.model.model_name,
        max_len=cfg.model.max_len,
    )

    model = TweetLightningModule(
        model_name=cfg.model.model_name,
        lr=cfg.training.lr,
    )

    tracking_uri = to_absolute_path(cfg.logging.tracking_uri)
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=tracking_uri,
        run_name=cfg.logging.run_name,
    )

    ckpt_dir = to_absolute_path(cfg.training.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=cfg.training.best_model,
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=cfg.training.save_top_k,
        save_last=False,
        auto_insert_metric_name=False,
    )

    output_dir = to_absolute_path(cfg.training.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=checkpoint_cb,
        default_root_dir=output_dir,
        logger=mlflow_logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
