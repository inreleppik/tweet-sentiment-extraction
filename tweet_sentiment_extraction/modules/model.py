import torch 
from torch import nn
from transformers import RobertaModel, RobertaConfig

class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        
        # грузим конфиг и включаем hidden_states
        config = RobertaConfig.from_pretrained("roberta-base")
        config.output_hidden_states = True

        # сама модель
        self.roberta = RobertaModel.from_pretrained("roberta-base", config=config)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0.0)

    def forward(self, input_ids, attention_mask):
        # вызываем модель с именованными аргументами
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # tuple тензоров: (embeddings, layer1, ..., layer12)
        hidden_states = outputs.hidden_states

        # берём последние 4 слоя и стакаем по новой оси: (4, batch, seq_len, hidden)
        x = torch.stack(hidden_states[-4:], dim=0)
        # среднее по этим 4 слоям → (batch, seq_len, hidden)
        x = x.mean(dim=0)

        x = self.dropout(x)
        x = self.fc(x)  # (batch, seq_len, 2)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits