import torch
from torch import nn
from tqdm.notebook import tqdm
import numpy as np

# 藉 start_idx, end_idx, offsets 取出 test 中的 selected_text
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        # 先取出指定範圍
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        # 確認是否需要加上空白做辨識
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text

# 建立 evaluation function - Jaccard index, 又稱Intersection over Union=一種測量在特定資料集中檢測相應物體準確度的一個標準
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    # 取聯集分之交集
    return float(len(c)) / (len(a) + len(b) - len(c))

# 計算 jaccard_score
def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    # 取出 機率最大的位置
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    
    # 此區取出預測區段文字，第一個條件判斷出有可能是整句文字的狀況
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
    
    # 取出正確對應語氣的文字
    true = get_selected_text(text, start_idx, end_idx, offsets)
    
    # 計算 jaccard_score
    return jaccard(true, pred)

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss



def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    # 使用 GPU
    model.cuda()

    # 根據訓練回數，每回訓練進行...
    for epoch in tqdm(range(num_epochs)):
        # 判斷當前階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            # 預設 loss 及 jaccard 為 0
            epoch_loss = 0.0
            epoch_jaccard = 0.0
            
            # 取出當前階段(train 或 val) 所使用的資料集，資料若是 torch tensor，在 GPU 訓練要轉成 GPU 使用的 Tesnor
            for data in tqdm((dataloaders_dict[phase])):
                ids = data['ids'].cuda()
                masks = data['masks'].cuda()
                tweet = data['tweet']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].cuda()
                end_idx = data['end_idx'].cuda()
                
                # 初始化 optimizer
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    
                    # 輸入 ids, masks 得到 model 輸出
                    start_logits, end_logits = model(ids, masks)
                    # 計算 loss
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    
                    # 在訓練階段要反向傳播且讓 optimizer 進行梯度下降
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # 計算各批訓練 loss 之總和，loss.item() 目的在於將 loss 取出成 python float 形式
                    epoch_loss += loss.item() * len(ids)
                    
                    # 以下步驟目的在於將 tensor 從 gpu 拿回 cpu 並且轉成 numpy array
                    # .cpu() 用於將 tensor 放回 cpu
                    # .detach() 用於阻斷反向傳播
                    # .numpy() 將 tensor 轉為 numpy array
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    
                    # 計算本回的總 jaccard 分數總合
                    for i in range(len(ids)):                        
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i], 
                            end_logits[i], 
                            offsets[i])
                        epoch_jaccard += jaccard_score
            
            # 平均 loss 及 jaccard
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)
            
            # 印出當前 Loss 及 jaccard
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
            
    # 儲存模型
    torch.save(model.state_dict(), filename)