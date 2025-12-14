import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tweet_sentiment_extraction.modules.dataset import TweetDataset
from tweet_sentiment_extraction.modules.module import TweetLightningModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    test_csv_path = to_absolute_path(cfg.data_loading.test_csv)
    test_df = pd.read_csv(test_csv_path)
    test_df["text"] = test_df["text"].astype(str)

    test_ds = TweetDataset(
        test_df,
        model_name=cfg.model.model_name,
        max_len=cfg.model.max_len,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
    )

    ckpt_dir = to_absolute_path(cfg.training.ckpt_dir)
    model_file = cfg.training.infer_model
    ckpt_path = os.path.join(ckpt_dir, model_file)

    model = TweetLightningModule.load_from_checkpoint(
        ckpt_path,
        model_name=cfg.model.model_name,
        lr=0.0,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    all_preds_batches = trainer.predict(model, dataloaders=test_loader)
    predictions = [p for batch in all_preds_batches for p in batch]

    res_dir = to_absolute_path(cfg.training.res_dir)
    res_file = cfg.training.res_file
    res_path = os.path.join(res_dir, res_file)
    os.makedirs(res_dir, exist_ok=True)

    test_df["selected_text"] = predictions
    test_df.to_csv(res_path, index=False)
    print(f"Saved {res_path}")


if __name__ == "__main__":
    main()
