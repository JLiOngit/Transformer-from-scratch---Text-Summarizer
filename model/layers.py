import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
import math
  

def positional_encoding(num_pos, dim_emb):
    positions = np.arange(num_pos)[:, np.newaxis]
    dims = np.arange(dim_emb)[np.new_axis, :]
    i = dims // 2
    angles = positions * (1 / np.power(10000, 2 * i / dim_emb))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = th.tensor(angles, dtype=th.float32).unsqueeze(0)
    return pos_encoding


class Embedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.embedding = nn.Embedding(vocab_size, dim_emb)

    def forward(self, x):
        _, _, vocab_size = x.shape
        assert vocab_size == self.vocab_size, "Vocabulary size mismatch"
        return self.embedding(x) * math.sqrt(self.dim_emb)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 num_heads,
                 dim_emb):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_emb
        assert dim_emb % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.dim_head = dim_emb // num_heads
        # Layers for linear projections
        self.input_layer = nn.Linear(dim_emb, 3*dim_emb)
        self.output_layer = nn.Linear(dim_emb, dim_emb)

    def forward(self, x_emb, mask=False):
        batch, length, dim_emb = x_emb.shape
        assert dim_emb == self.dim_head, "Embedding dimension mismatch"
        input_shape = batch, self.num_heads, length, 3*self.dim_head
        qkv = self.input_layer(x_emb).view(input_shape)
        q, k, v = th.chunk(qkv, 3, dim=-1)
        scores = th.matmul(q, k.transpose(-1, -2) / math.sqrt(self.dim_head))
        if mask:
            mask = th.ones_like(scores, dtype=th.bool).tril(1)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        attention = th.matmul(weights, v).transpose(1,2).reshape(batch, length, dim_emb)
        output = self.output_layer(attention)
        return output
    

class CrossAttention(nn.Module):

    def __init__(self,
                 num_heads,
                 dim_emb,
                 dim_cross):
         super().__init__()
         self.num_heads = num_heads
         self.dim_emb = dim_emb
         assert dim_emb % num_heads == 0, "Embedding dimension must be divisible by number of heads"
         self.dim_head = dim_emb // num_heads
         # Layers for linear projections
         self.query_layer = nn.Linear(dim_emb, dim_emb)
         self.key_layer = nn.Linear(dim_cross, dim_emb)
         self.value_layer = nn.Linear(dim_cross, dim_emb)
         self.output_layer = nn.Linear(dim_emb, dim_emb)

    def forward(self, x_emb, encoder_output):
        batch, length, dim_emb = x_emb.shape
        assert dim_emb == self.dim_emb, "Embedding dimension mismatch"
        input_shape = batch, self.num_heads, length, self.dim_head
        q = self.query_layer(x_emb).view(input_shape)
        k = self.key_layer(encoder_output).view(input_shape)
        v = self.value_layer(encoder_output).view(input_shape)
        scores = th.matmul(q, k.transpose(-1, -2) / math.sqrt(self.dim_head))
        weights = F.softmax(scores, dim=-1)
        attention = th.matmul(weights, v).transpose(1,2).reshape(batch, length, dim_emb)
        output = self.output_layer(attention)
        return output


class FullyConnected(nn.Module):

    def __init__(self,
                 dim_emb,
                 dim_ff):
        super().__init__()
        self.dim_output = dim_emb
        self.dim_ff = dim_ff
        # Layers of the feed-forward network
        self.in_layer = nn.Sequential([
            nn.Linear(dim_emb, dim_ff),
            nn.RELU(),
            nn.Dropout()
            ])
        self.out_layer = nn.Linear(dim_ff, dim_emb)
    
    def forward(self, x_emb):
        _, _, dim_emb = x_emb.shape
        assert dim_emb == self.dim_output, "Embedding dimension mismatch"
        return self.out_layer(self.in_layer(x_emb))


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
    

    