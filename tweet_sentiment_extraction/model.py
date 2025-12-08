import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from config import MODEL_NAME

class TweetModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = RobertaConfig.from_pretrained(MODEL_NAME)
        config.output_hidden_states = True

        self.roberta = RobertaModel.from_pretrained(MODEL_NAME, config=config)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0.0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.hidden_states
        x = hidden_states[-4:]
        x = sum(x) / 4.0

        x = self.dropout(x)
        x = self.fc(x)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    return start_loss + end_loss