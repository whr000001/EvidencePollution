from coattention import CoAttention
from hypComEnc import HypComEnc
from hypPostEnc import HypPostEnc
from utils.layers.hyp_layers import *
from utils.manifolds import PoincareBall


class Hyphen(nn.Module):
    def __init__(self, embedding_matrix, word_hidden_size, sent_hidden_size, device,
                 graph_hidden, num_classes, latent_dim, graph_glove_dim):
        super(Hyphen, self).__init__()

        manifold = PoincareBall()

        self.content_encoder = HypPostEnc(word_hidden_size, sent_hidden_size, num_classes,
                                          embedding_matrix, device)
        self.comment_encoder = HypComEnc(graph_glove_dim, graph_hidden,
                                         device=device, manifold=manifold)
        self.co_attention = CoAttention(latent_dim, manifold=manifold)

        self.cls = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, content, comment, subgraphs, label):
        _, content_embedding = self.content_encoder(content)
        comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)

        coatten, As, Ac = self.co_attention(content_embedding, comment_embedding)
        pred = self.cls(coatten)

        loss = self.loss_fn(pred, label)
        return pred, loss
