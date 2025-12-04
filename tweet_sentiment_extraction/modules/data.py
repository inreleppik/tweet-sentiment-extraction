import pandas as pd
import torch
from transformers import RobertaTokenizerFast

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df

        # fast-токенизатор, чтобы был return_offsets_mapping
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            "roberta-base",
            add_prefix_space=True,
        )

    # 賦予此 Class 用 index 取值的能力， e.g. TweetDataset[1]
    def __getitem__(self, index):
        # 建立空的 dictionary
        data = {}
        # iloc:用index位置來取我們要的資料
        row = self.df.iloc[index] 
        # 使用 class 函式 get_input_data 根據 index row 取值且放入剛剛的 data dictionary
        ids, masks, tweet, offsets = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks #，由於 padding 會替不等長的句子們補0 ， 這時候利用masks就可以標註出非 0 的區域，也就是讓模型不被 padding 補的 0 影響判斷。
        data['tweet'] = tweet
        data['offsets'] = offsets #是一個表示 該單詞於句子的起始位置 結束位置的元組
        
        # 若 labeled 不為空集合則執行
        if self.labeled:
            # 使用 class 函式 get_target_idx, 額外針對目標取出 start_idx, end_idx 
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
            
        # 回傳 data dictionary
        return data
    
    # 定義針對此 class 呼叫 python 內建函式 len 的時候的回傳值
    def __len__(self):
        return len(self.df)
    
    # 傳入一列資料，回傳 ids, masks, tweet, offsets 四個變數
    def get_input_data(self, row): 
        # как раньше
        tweet = " " + " ".join(row.text.lower().split())

        # токенизируем сам твит БЕЗ спец-токенов, чтобы offset'ы соответствовали строке
        encoding = self.tokenizer(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,   # обрежем сами ниже, чтобы учесть спец-токены
        )

        tweet_ids = encoding["input_ids"]
        tweet_offsets = encoding["offset_mapping"]

        # токенизируем sentiment (positive / negative / neutral)
        sent_enc = self.tokenizer(
            row.sentiment,
            add_special_tokens=False
        )
        sentiment_ids = sent_enc["input_ids"]

        # ручная сборка input_ids, как в исходном коде
        # у RoBERTa: <s>=0, </s>=2, pad=1
        ids = [0] + sentiment_ids + [2, 2] + tweet_ids + [2]

        # offsets: на всё до твита ставим заглушки (0,0)
        prefix_len = 1 + len(sentiment_ids) + 2  # <s> + sentiment + </s></s>
        offsets = [(0, 0)] * prefix_len + list(tweet_offsets) + [(0, 0)]

        # если слишком длинно — обрежем
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            offsets = offsets[:self.max_len]

        # паддинг, если коротко
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len           # pad token id = 1
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)

        return ids, masks, tweet, offsets
    
    '''
    此資料集的目標是指出該列 Text 能夠判斷語氣的部份, 
    放置於 train 資料集的 selected_text 欄位
    '''
    def get_target_idx(self, row, tweet, offsets):
        # 同上 text 處理方法
        selected_text = " " +  " ".join(row.selected_text.lower().split())
        
        # 取出 selected_text 的長度
        len_st = len(selected_text) - 1
        # 建立 text 之 index 用 #?
        idx0 = None
        idx1 = None
        

        # 在 e == selected_text[1] , 也就是與 selected_text 開頭的單詞相同的句子的集合內  enumerate=利用它可以同時獲得索引和值
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            # 若 " " + tweet[ind: ind+len_st] 的組合 和 selected_text 一樣
            if " " + tweet[ind: ind+len_st] == selected_text:
                # 設定 idx0 為起始點, idx1 為終止點
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        
        # 先以 len(tweet) 個 [0] 初始化 char_targets
        char_targets = [0] * len(tweet)
        # 若有成功取出 idx0 及 idx1
        if idx0 != None and idx1 != None:
            # 將 char_targets 對應 tweet 的 selected_text 位置 (idx0 ~ idx1 的範圍) 設為 1
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        # 藉 offset 製造 target_idx 做訓練使用
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            # 若有發現 char_targets 中 範圍 offset1 至 offset2 的和大於 0 (代表有值)，
            # 則將其 index 放入 target_idx
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        # 起始 idx 為 target_idx 中第一個，終止 idx 則為最後一個
        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        
        return start_idx, end_idx

'''
傳入 dataframe, 分割後之 train 及 val 對應的 idx, 及預設為 8 的 batch_size
回傳有 train 及 val DataLoader 的 dictionary
'''
def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    # 藉 train_idx 及 val_idx 將 dataframe 分割成訓練及驗證 dataframe
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    
    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df), 
        batch_size=batch_size, 
        shuffle=True,  # 打亂排序 
        num_workers=2, # 以兩個 子行程處理
        drop_last=True) # 當資料集 batch 無法均分時，捨棄最後一個不完整的 batch

    # 要注意不要打亂排序避免 idx 錯亂
    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)
    
    # 用 dict 儲存兩個 Loader, 並且加上對應的 Key
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

'''
傳入 dataframe, 及預設為 32 的 batch_size
回傳 test 資料集使用的 Loader 
'''
def get_test_loader(df, batch_size=8):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), 
        batch_size=batch_size, 
        shuffle=False, # 找出答案用, 所以不打亂順序
        num_workers=2)  # 以兩個 子行程 處理    
    return loader