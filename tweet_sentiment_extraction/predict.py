import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import TEST_CSV, OUTPUT_DIR, BATCH_SIZE
from dataset import TweetDataset
from metrics import get_selected_text
from module import TweetLightningModule

def main():
    test_df = pd.read_csv(TEST_CSV)
    test_df["text"] = test_df["text"].astype(str)

    test_ds = TweetDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    ckpt_path = f"{OUTPUT_DIR}/roberta_single_split.ckpt"
    pl_model = TweetLightningModule.load_from_checkpoint(ckpt_path, lr=0.0)
    pl_model.eval()
    pl_model.cuda()
    model = pl_model.model

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["ids"].cuda()
            masks = batch["masks"].cuda()
            tweet = batch["tweet"]
            offsets = batch["offsets"].numpy()

            start_logits, end_logits = model(ids, masks)
            start_logits = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_logits = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(len(ids)):
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

    test_df["selected_text"] = predictions
    test_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()