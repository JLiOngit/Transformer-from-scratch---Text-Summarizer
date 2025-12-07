import numpy as np
import torch as th
from torch import nn
from layers import *


def positional_encoding(num_pos, dim_emb):
    positions = np.arange(num_pos)[:, np.newaxis]
    dims = np.arange(dim_emb)[np.new_axis, :]
    i = dims // 2
    angles = positions * (1 / np.power(10000, 2 * i / dim_emb))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = th.tensor(angles, dtype=th.float32).unsqueeze(0)
    return pos_encoding

class InputEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim_emb):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.embedding = nn.Embedding(vocab_size, dim_emb)

    def forward(self, x):
        return self.embedding(x)


class Encoder(nn.Module):

    def __init__(self,
                 num_layers,
                 vocab_size,
                 dim_emb,
                 num_heads,
                 dim_ff):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        self.input_embedding = InputEmbedding(vocab_size, dim_emb)
        self.multihead_attentions = MultiHeadAttention(num_heads, dim_emb)
        self.layer_norm = nn.LayerNorm()
        self.fully_connected = FullyConnected(dim_emb, dim_ff)
    
    def forward(self, x):

        batch, length, vocab_size = x.shape
        x_emb = self.input_embedding(x)
        positional_encoding = positional_encoding(length, self.dim_emb).to(x.device)
        x_emb = x_emb + positional_encoding
        for _ in range(self.num_layers):
            x_emb = ResidualConnection(self.multihead_attentions)(x_emb)
            x_emb = self.layer_norm(x_emb)
            x_emb = ResidualConnection(self.fully_connected)(x_emb)
            x_emb = self.layer_norm(x_emb)
        return x_emb

