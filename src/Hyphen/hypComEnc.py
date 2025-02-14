import dgl
from utils.layers.hyp_layers import *


class HypComEnc(nn.Module):
    def __init__(self, in_dim, hidden_dim, device, manifold):
        super(HypComEnc, self).__init__()
        self.manifold = manifold
        self.c = True
        self.conv1 = HGCNLayer(self.manifold, in_dim, hidden_dim, c_in=self.c, c_out=self.c,
                               act=torch.tanh, dropout=0.1, use_bias=True)
        self.conv2 = HGCNLayer(self.manifold, hidden_dim, hidden_dim, c_in=self.c, c_out=self.c,
                               act=torch.tanh, dropout=0.1, use_bias=True)
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_comment_count = 10

    def forward(self, g, h, subgraphs):
        # Apply graph convolution and activation.
        adj = g.adj().to(self.device)  # finding the adjacency matrix
        inp = h.to(self.device)  # converting to sparse tensor
        inp = [self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(i, c=self.c), c=self.c), c=self.c).
               unsqueeze(0) for i in inp]
        inp = torch.cat(inp, dim=0)
        inp = inp.to(dtype=torch.float)
        adj = adj.to_dense()
        out, adj = self.conv1((inp, adj))
        out, adj = self.conv2((out, adj))
        h = out.to_dense()  # converting back to dense
        h = self.manifold.logmap0(self.manifold.proj(h, c=self.c), c=self.c)
        # map h (which is in poincare space/euclidean) to tangential space to aggregate the node representations
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            un_batched = dgl.unbatch(g)
            batch_agg = []
            for batch_idx in range(len(un_batched)):
                agg = []
                for node_list in subgraphs[batch_idx]:
                    sub = dgl.node_subgraph(un_batched[batch_idx], node_list)
                    hg = dgl.mean_nodes(sub, 'h')
                    agg.append(torch.squeeze(hg).unsqueeze(0))
                if len(agg) >= self.max_comment_count:
                    agg = agg[:self.max_comment_count]
                    agg = torch.cat([i.float() for i in agg], dim=0)
                else:
                    padding = torch.zeros((self.max_comment_count - len(agg), self.hidden_dim), dtype=torch.float32,
                                          requires_grad=True).to(self.device)
                    without_padding = torch.cat([i.float() for i in agg], dim=0)
                    agg = torch.cat([without_padding, padding], dim=0)
                agg = self.manifold.proj(self.manifold.expmap0(agg, c=self.c), c=self.c)
                batch_agg.append(agg.unsqueeze(0))
            ret = torch.cat(batch_agg, dim=0)
            return ret
