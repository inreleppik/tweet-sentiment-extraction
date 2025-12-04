# 讀入測試(輸出答案)用 csv
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
# 將 text 內容轉型為 string
test_df['text'] = test_df['text'].astype(str)
# 取得 test 用 dataloader
test_loader = get_test_loader(test_df)

# 初始化
predictions = []
models = []

# 讀出每個 fold 訓練出的 Model 並且放到 models 中
for fold in range(skf.n_splits):
    model = TweetModel()
    model.cuda()
    model.load_state_dict(torch.load(f'roberta_fold{fold+1}.pth'))
    model.eval()
    models.append(model)

for data in test_loader:
    #資料若是 torch tensor，在 CPU 用要轉成 GPU 使用的 Tesnor
    ids = data['ids'].cuda()
    masks = data['masks'].cuda()
    tweet = data['tweet']
    offsets = data['offsets'].numpy()

    start_logits = []
    end_logits = []
    # 運算出每個 fold 訓練下的輸出結果，並且放回 cpu，阻斷反向傳播，再轉成 numpy array
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
            end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())
    # 沿著維度 0 號取平均
    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)
    for i in range(len(ids)):    
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        # 取出預測區段文字，有可能是整句
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        # 放入 predictions
        predictions.append(pred)