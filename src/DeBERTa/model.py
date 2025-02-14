import torch
import torch.nn as nn
from transformers import AutoModel
from torch_scatter import scatter


class MyModel(nn.Module):
    def __init__(self, num_class, hidden_dim=512):
        super().__init__()
        self.lm = AutoModel.from_pretrained('/data02/whr/resources/DeBERTa-v3', local_files_only=True)
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
        self.content_fc = nn.Linear(1024, hidden_dim // 2)
        self.comment_fc = nn.Linear(1024, hidden_dim // 2)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class)
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
        content_reps = self._get_reps(data['content'])

        comment_batch = data['comment_batch']
        comment_reps = self._get_reps(data['comment'])
        size = int(comment_batch.max().item() + 1)
        comment_reps = scatter(comment_reps, comment_batch, dim=-2, dim_size=size, reduce='mean')

        content_reps = self.content_fc(content_reps)
        comment_reps = self.comment_fc(comment_reps)
        reps = torch.cat([content_reps, comment_reps], dim=-1)

        preds = self.cls(reps)
        loss = self.cls_loss(preds, data['label'])
        return preds, loss


class MyModelWithoutComments(nn.Module):
    def __init__(self, num_class, hidden_dim=512):
        super().__init__()
        self.lm = AutoModel.from_pretrained('/data02/whr/resources/DeBERTa-v3', local_files_only=True)
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
        self.content_fc = nn.Linear(1024, hidden_dim)
        # self.comment_fc = nn.Linear(1024, hidden_dim // 2)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class)
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
        content_reps = self._get_reps(data['content'])

        # comment_batch = data['comment_batch']
        # comment_reps = self._get_reps(data['comment'])
        # size = int(comment_batch.max().item() + 1)
        # comment_reps = scatter(comment_reps, comment_batch, dim=-2, dim_size=size, reduce='mean')

        content_reps = self.content_fc(content_reps)
        # comment_reps = self.comment_fc(comment_reps)
        reps = content_reps

        preds = self.cls(reps)
        loss = self.cls_loss(preds, data['label'])
        return preds, loss
