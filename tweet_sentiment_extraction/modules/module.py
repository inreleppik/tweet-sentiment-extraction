import numpy as np
import pytorch_lightning as pl
import torch

from tweet_sentiment_extraction.modules.metrics import TextSpanJaccard, get_selected_text
from tweet_sentiment_extraction.modules.model import TweetModel, loss_fn


class TweetLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = TweetModel(model_name=model_name)
        self.lr = lr
        self.train_jaccard = TextSpanJaccard()
        self.val_jaccard = TextSpanJaccard()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        ids = batch["ids"]
        masks = batch["masks"]
        start_idx = batch["start_idx"]
        end_idx = batch["end_idx"]

        start_logits, end_logits = self(ids, masks)
        loss = loss_fn(start_logits, end_logits, start_idx, end_idx)

        self.train_jaccard.update(batch, start_logits, end_logits)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_jaccard",
            self.train_jaccard,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        masks = batch["masks"]
        start_idx = batch["start_idx"]
        end_idx = batch["end_idx"]

        start_logits, end_logits = self(ids, masks)
        loss = loss_fn(start_logits, end_logits, start_idx, end_idx)

        self.val_jaccard.update(batch, start_logits, end_logits)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_jaccard",
            self.val_jaccard,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ids = batch["ids"]
        masks = batch["masks"]
        tweet = batch["tweet"]
        offsets = batch["offsets"].detach().cpu().numpy()

        start_logits, end_logits = self(ids, masks)
        start_probs = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        preds = []
        for i in range(len(tweet)):
            start_pred = np.argmax(start_probs[i])
            end_pred = np.argmax(end_probs[i])

            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            preds.append(pred)

        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
