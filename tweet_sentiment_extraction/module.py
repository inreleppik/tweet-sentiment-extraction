import torch
import pytorch_lightning as pl

from model import TweetModel, loss_fn
from metrics import compute_jaccard_score


class TweetLightningModule(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = TweetModel()
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        ids = batch["ids"]
        masks = batch["masks"]
        start_idx = batch["start_idx"]
        end_idx = batch["end_idx"]

        start_logits, end_logits = self(ids, masks)
        loss = loss_fn(start_logits, end_logits, start_idx, end_idx)

        # === Jaccard ===
        jaccard = self._compute_batch_jaccard(batch, start_logits, end_logits)

        # логируем
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_jaccard", jaccard, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        masks = batch["masks"]
        start_idx = batch["start_idx"]
        end_idx = batch["end_idx"]

        start_logits, end_logits = self(ids, masks)
        loss = loss_fn(start_logits, end_logits, start_idx, end_idx)

        jaccard = self._compute_batch_jaccard(batch, start_logits, end_logits)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_jaccard", jaccard, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )

    def _compute_batch_jaccard(self, batch, start_logits, end_logits):
        """
        Вычисление среднего Jaccard по батчу.
        """

        # всё приводим к numpy
        start_logits = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_logits = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        start_true = batch["start_idx"].detach().cpu().numpy()
        end_true = batch["end_idx"].detach().cpu().numpy()
        offsets = batch["offsets"].detach().cpu().numpy()
        tweets = batch["tweet"]

        scores = []
        for i in range(len(tweets)):
            score = compute_jaccard_score(
                tweets[i],
                start_true[i],
                end_true[i],
                start_logits[i],
                end_logits[i],
                offsets[i],
            )
            scores.append(score)

        return sum(scores) / len(scores)