from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import TweetDataset
from config import BATCH_SIZE, MAX_LEN

class TweetDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, batch_size=BATCH_SIZE, max_len=MAX_LEN, num_workers=2):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TweetDataset(self.train_df, max_len=self.max_len)
        self.val_ds = TweetDataset(self.val_df, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )