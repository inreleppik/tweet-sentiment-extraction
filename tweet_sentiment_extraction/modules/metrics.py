import numpy as np
import torch
from torchmetrics import Metric


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) + len(b) - len(c) == 0:
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)

    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)


class TextSpanJaccard(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_jaccard", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch, start_logits, end_logits):
        start_probs = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        start_true = batch["start_idx"].detach().cpu().numpy()
        end_true = batch["end_idx"].detach().cpu().numpy()
        offsets = batch["offsets"].detach().cpu().numpy()
        tweets = batch["tweet"]

        total = 0.0
        n = len(tweets)
        for i in range(n):
            score = compute_jaccard_score(
                tweets[i],
                start_true[i],
                end_true[i],
                start_probs[i],
                end_probs[i],
                offsets[i],
            )
            total += score

        self.sum_jaccard += torch.tensor(total, dtype=torch.float32, device=self.sum_jaccard.device)
        self.n_samples += torch.tensor(n, dtype=torch.long, device=self.n_samples.device)

    def compute(self):
        if self.n_samples == 0:
            return torch.tensor(0.0, device=self.sum_jaccard.device)
        return self.sum_jaccard / self.n_samples
