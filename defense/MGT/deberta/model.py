import torch
import torch.nn as nn
from transformers import AutoModel
from torch_scatter import scatter


class MyModel(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lm = AutoModel.from_pretrained('microsoft/deberta-v3-base')  # enter your own path here
        for name, param in self.lm.named_parameters():
            param.requires_grad = True
        self.cls = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.cls_loss = nn.CrossEntropyLoss()

    def _get_reps(self, inputs):
        reps = self.lm(**inputs).last_hidden_state
        attention_mask = inputs['attention_mask']
        reps = torch.einsum('ijk,ij->ijk', reps, attention_mask)
        reps = torch.sum(reps, dim=1)
        attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
        reps = reps / attention_mask
        return reps

    def forward(self, data):
        reps = self._get_reps(data['text'])
        preds = self.cls(reps)
        loss = self.cls_loss(preds, data['label'])
        return preds, loss
