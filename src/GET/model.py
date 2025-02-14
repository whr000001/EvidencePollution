import torch
import torch.nn as nn
from transformers import DebertaV2Tokenizer, DebertaV2Model
from torch_geometric.nn.conv import GCNConv
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from einops import repeat
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as func
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import subgraph


def split_graph(score, batch):
    from torch_geometric.utils import degree

    def sparse_sort(src, index, dim=0, descending=False, eps=1e-12):
        f_src = src.float()
        f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
        norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
        perm = norm.argsort(dim=dim, descending=descending)
        return src[perm], perm

    _, node_perm = sparse_sort(score, batch, descending=True)
    num_nodes = degree(batch, dtype=torch.long)
    k = (0.2 * num_nodes.to(float)).ceil().to(torch.long)
    start_indices = torch.cat([torch.zeros((1,), device=score.device, dtype=torch.long), num_nodes.cumsum(0)])
    mask = [torch.arange(k[i], dtype=torch.long, device=score.device) + start_indices[i] for i in
            range(len(num_nodes))]
    mask = torch.cat(mask, dim=0)
    mask = torch.zeros_like(batch, device=batch.device).index_fill(0, mask, 1).bool()
    sub_node = node_perm[mask]
    return sub_node


class MyModel(nn.Module):
    def __init__(self, num_class, device):
        super().__init__()
        self.num_class = num_class
        self.device = device

        embedding = torch.load('../../data/word_vecs.pt')
        embedding_default = torch.zeros(1, 300, dtype=torch.float)
        self.word_embedding = nn.Embedding(embedding.shape[0] + 1, embedding.shape[1])
        self.word_embedding.data = torch.cat([embedding, embedding_default], dim=0)
        self.word_embedding.requires_grad_ = False

        self.content_convs = nn.ModuleList([GCNConv(300, 512), GCNConv(512, 512)])
        self.content_pooling = global_mean_pool

        self.comment_trans = GCNConv(300, 512)
        self.score_pred = nn.Linear(512, 1)
        self.comment_encoder = GCNConv(300, 512)

        self.node_attn = nn.MultiheadAttention(512, 4, 0.5)
        self.comment_attn = nn.MultiheadAttention(512, 4, 0.5)

        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.cls = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        content_graph = batch['content_graph']
        content_reps = self.word_embedding(content_graph.x)
        # print(news_reps.shape)
        for conv in self.content_convs:
            content_reps = conv(content_reps, content_graph.edge_index)
            content_reps = self.dropout(self.act_fn(content_reps))
        news_reps = self.content_pooling(content_reps, content_graph.batch)

        comments_graph = batch['comments_graph']
        comments_reps = self.word_embedding(comments_graph.x)
        h = self.comment_trans(comments_reps, comments_graph.edge_index)
        score = self.score_pred(h).view(-1)
        keep_node_mask = split_graph(score, comments_graph.batch)
        comments_edge_index, _ = subgraph(keep_node_mask, comments_graph.edge_index, relabel_nodes=True)
        comments_reps = comments_reps[keep_node_mask]
        comments_graph_batch = comments_graph.batch[keep_node_mask]
        comments_reps = self.act_fn(self.dropout(self.comment_encoder(comments_reps, comments_edge_index)))
        comments_reps_each = [[] for _ in range(batch['batch_size'])]
        comments_batch = batch['comments_batch']
        for index in range(comments_graph.num_graphs):
            batch_index = comments_batch[index].item()
            mask = torch.eq(comments_graph_batch, index)
            comment_node_reps = comments_reps[mask]
            news_reps_each = news_reps[batch_index].unsqueeze(0)
            out, _ = self.node_attn(news_reps_each, comment_node_reps, comment_node_reps)
            comments_reps_each[batch_index].append(out)
        final_comment = []
        for batch_index in range(batch['batch_size']):
            each_comment = comments_reps_each[batch_index]
            each_comment = torch.cat(each_comment, dim=0)
            news_reps_each = news_reps[batch_index].unsqueeze(0)
            out, _ = self.comment_attn(news_reps_each, each_comment, each_comment)
            final_comment.append(out)
        final_comment = torch.cat(final_comment, dim=0)
        reps = torch.cat([news_reps, final_comment], dim=-1)
        pred = self.cls(reps)

        loss = self.loss_fn(pred, batch['label'].to(self.device))

        return pred, loss
