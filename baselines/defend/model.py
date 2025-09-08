import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
word_vec = torch.load('../data/word_vecs.pt')
embedding_default = torch.zeros(1, 300, dtype=torch.float)
embedding_layer = nn.Embedding(word_vec.shape[0] + 1, word_vec.shape[1])
embedding_layer.weight.data = torch.cat([word_vec, embedding_default], dim=0)
embedding_layer.weight.requires_grad = False
embedding_layer.to(device)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, eps=1e-5):
        super(AttentionLayer, self).__init__()
        self.epsilon = eps
        self.W = nn.Parameter(torch.randn((in_dim, hidden_dim)))
        self.b = nn.Parameter(torch.randn(hidden_dim))
        self.u = nn.Parameter(torch.randn((hidden_dim, 1)))

    def forward(self, x):
        uit = torch.tanh(torch.matmul(x, self.W) + self.b)
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)
        ait = ait / (torch.sum(ait, dim=1, keepdim=True) + self.epsilon)
        ait = torch.unsqueeze(ait, -1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, dim=1)
        return output


class CoAttention(nn.Module):
    def __init__(self, hidden_dim, k):
        super(CoAttention, self).__init__()
        self.Wl = nn.Parameter(torch.randn((hidden_dim, hidden_dim)))
        self.Wc = nn.Parameter(torch.randn((k, hidden_dim)))
        self.Ws = nn.Parameter(torch.randn((k, hidden_dim)))
        self.whs = nn.Parameter(torch.randn((1, k)))
        self.whc = nn.Parameter(torch.randn((1, k)))

    def forward(self, sentence_rep, comment_rep):
        sentence_rep_trans = sentence_rep.transpose(2, 1)
        comment_rep_trans = comment_rep.transpose(2, 1)
        L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))
        L_trans = L.transpose(2, 1)
        Hs = torch.tanh(
            torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        Hc = torch.tanh(
            torch.matmul(self.Wc, comment_rep_trans) + torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        As = F.softmax(torch.matmul(self.whs, Hs), dim=2)
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)
        As = As.transpose(2, 1)
        Ac = Ac.transpose(2, 1)
        co_s = torch.matmul(sentence_rep_trans, As)
        co_c = torch.matmul(comment_rep_trans, Ac)
        co_sc = torch.cat([co_s, co_c], dim=1)
        return torch.squeeze(co_sc, -1)


class DEFEND(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, k, num_classes):
        super(DEFEND, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.sentence_encoder = nn.GRU(embedding_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.comment_encoder = nn.GRU(embedding_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.content_encoder = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(in_dim=hidden_dim, hidden_dim=hidden_dim)
        self.co_attention = CoAttention(hidden_dim=hidden_dim, k=k)

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def init_hidden(self, batch_size):
        # return (torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
        #         torch.randn(2, batch_size, self.hidden_dim // 2).to(device))
        return torch.zeros(2, batch_size, self.hidden_dim // 2).to(device)

    def forward(self, batch):
        content = batch['content'].to(device)
        comment = batch['comment'].to(device)
        batch_size = content.shape[0]
        # print(content.shape)
        embedded_content = embedding_layer(content)
        embedded_comment = embedding_layer(comment)
        embedded_comment = embedded_comment.transpose(1, 0)
        embedded_content = embedded_content.transpose(1, 0)
        # print(embedded_content.shape)
        xa_cache = []
        xc_cache = []
        for sentence, comment in zip(embedded_content, embedded_comment):
            hidden = self.init_hidden(batch_size)
            x1, _ = self.sentence_encoder(sentence, hidden)
            xa = self.attention(x1)
            hidden = self.init_hidden(batch_size)
            x2, _ = self.comment_encoder(comment, hidden)
            xc = self.attention(x2)
            xa_cache.append(xa)
            xc_cache.append(xc)
        xa = torch.stack(xa_cache)
        xc = torch.stack(xc_cache)
        xa = xa.transpose(1, 0)
        xc = xc.transpose(1, 0)
        hidden = self.init_hidden(batch_size)
        x3, _ = self.content_encoder(xa, hidden)
        # if torch.isnan(x3).any().item():
        #     print('x3')
        #     input()
        # if torch.isnan(xc).any().item():
        #     print('xc')
        #     input()
        co_attn = self.co_attention(x3, xc)
        # if torch.isnan(co_attn).any().item():
        #     print('co_attn')
        #     input()

        pred = self.cls(co_attn)

        loss = self.loss_fn(pred, batch['label'].to(device))

        return pred, loss
