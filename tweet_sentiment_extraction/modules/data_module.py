from torch.utils.data import DataLoader
import pytorch_lightning as pl

from tweet_sentiment_extraction.modules.dataset import TweetDataset

class TweetDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, batch_size: int, max_len: int, model_name:str, num_workers=2):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.max_len = max_len
        self.model_name = model_name
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = TweetDataset(self.train_df, model_name=self.model_name, max_len=self.max_len)
        self.val_ds = TweetDataset(self.val_df, model_name=self.model_name, max_len=self.max_len)

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
    
class TweetInferDataModule(pl.LightningDataModule):
    def __init__(self, df, batch_size: int):
        super().__init__()
        self.df = df
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.ds = TweetDataset(self.df)

    def predict_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=False, num_workers=2)