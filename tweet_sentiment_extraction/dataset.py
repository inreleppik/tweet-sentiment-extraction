import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from config import MODEL_NAME, MAX_LEN

class TweetDataset(Dataset):
    def __init__(self, df, max_len: int = MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = 'selected_text' in df.columns

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            MODEL_NAME,
            add_prefix_space=True,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        ids, masks, tweet, offsets = self.get_input_data(row)

        data = {
            "ids": ids,
            "masks": masks,
            "tweet": tweet,
            "offsets": offsets,
        }

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
            data["end_idx"] = torch.tensor(end_idx, dtype=torch.long)

        return data

    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())

        encoding = self.tokenizer(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        tweet_ids = encoding["input_ids"]
        tweet_offsets = encoding["offset_mapping"]

        sent_enc = self.tokenizer(
            row.sentiment,
            add_special_tokens=False,
        )
        sentiment_ids = sent_enc["input_ids"]

        # <s> sentiment </s></s> tweet </s>
        ids = [0] + sentiment_ids + [2, 2] + tweet_ids + [2]

        prefix_len = 1 + len(sentiment_ids) + 2
        offsets = [(0, 0)] * prefix_len + list(tweet_offsets) + [(0, 0)]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            offsets = offsets[:self.max_len]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != 1).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())
        len_st = len(selected_text) - 1

        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx