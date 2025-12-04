import os
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from config import SEED, TRAIN_CSV, NUM_EPOCHS, OUTPUT_DIR, BATCH_SIZE, LR
from utils import seed_everything
from data_module import TweetDataModule
from module import TweetLightningModule

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seed_everything(SEED)
    pl.seed_everything(SEED, workers=True)

    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].astype(str)
    df["selected_text"] = df["selected_text"].astype(str)

    # один сплит с сохранением баланса по sentiment
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["sentiment"],
    )

    dm = TweetDataModule(train_df, val_df, batch_size=BATCH_SIZE)
    model = TweetLightningModule(lr=LR)

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)

    ckpt_path = os.path.join(OUTPUT_DIR, "roberta_single_split.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved: {ckpt_path}")

if __name__ == "__main__":
    main()