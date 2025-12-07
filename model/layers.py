import torch as th
from torch import nn
from torch.nn import functional as F
import math
  

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 num_heads,
                 dim_emb):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_emb
        try:
            assert dim_emb % num_heads == 0
            self.dim_head = dim_emb // num_heads
        except AssertionError:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        self.input_layer = nn.Linear(dim_emb, 3*dim_emb)
        self.output_layer = nn.Linear(dim_emb, dim_emb)

    def forward(self, x, mask=None):
        batch, length, dim_emb = x.shape
        qkv = nn.Linear(dim_emb, 3* dim_emb)(x)
        qkv = qkv.view(batch, self.num_heads, length, 3*self.dim_head)
        q, k, v = th.chunk(qkv, 3, dim=-1)
        scores = th.matmul(q, k.transpose(-1, -2) / math.sqrt(self.dim_head))
        if mask is not None:
            mask = th.ones_like(scores, dtype=th.bool).tril(1)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attention = th.matmul(weights, v).transpose(1,2).reshape(batch, length, dim_emb)
        output = nn.Linear(dim_emb, dim_emb)(attention)
        return output
    

class FullyConnected(nn.Module):

    def __init__(self,
                 dim_emb,
                 dim_ff):
        super().__init__()
        self.dim_output = dim_emb
        self.dim_ff = dim_ff
        self.in_layer(
            nn.Linear(dim_emb, dim_ff),
            nn.RELU(),
            nn.Dropout()
        )
        self.out_layer = nn.Linear(dim_ff, dim_emb)
    
    def forward(self, x):
        return self.out_layer(self.in_layer(x))


class ResidualConnection(nn.Module):

    def __init__(self,
                 sublayer):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout()

    def forward(self, x, *args, **kwargs):
        sublayer_output = self.sublayer(x, *args, **kwargs)
        output = x + self.dropout(sublayer_output)
        return output
    

    