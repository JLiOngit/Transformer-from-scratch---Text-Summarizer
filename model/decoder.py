import math
import torch as th
from torch import nn
from layers import *


class DecoderBlock(nn.Module):

    def __init__(self,
                 dim_emb,
                 dim_cross,
                 num_heads,
                 dim_ff):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        # Components of the decoder block
        self.masked_multihead_attentions = MultiHeadAttention(num_heads, dim_emb, mask=True)
        self.layer_norm_1 = nn.LayerNorm()
        self.cross_attentions = CrossAttention(num_heads, dim_emb, dim_cross)
        self.layer_norm_2 = nn.LayerNorm()
        self.fully_connected = FullyConnected(dim_emb, dim_ff)
        self.layer_norm_3 = nn.LayerNorm()

    def forward(self, x_emb, encoder_output):
        _, _, dim_emb = x_emb.shape
        assert dim_emb == self.dim_emb, "Embedding dimension mismatch"
        output = ResidualConnection(self.masked_multihead_attentions)(x_emb)
        output = self.layer_norm_1(output)
        output = ResidualConnection(self.cross_attentions)(output, encoder_output)
        output = self.layer_norm_2(output)
        output = ResidualConnection(self.fully_connected)(output)
        output = self.layer_norm_3(output)
        return output
    

class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim_emb,
                 dim_cross,
                 num_heads,
                 dim_ff,
                 num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.dim_cross = dim_cross
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        # Components of the decoder
        self.output_embedding = Embedding(vocab_size, dim_emb)
        self.decoder_block = DecoderBlock(dim_emb, dim_cross, num_heads, dim_ff)

    def forward(self, x, encoder_output):
        batch, length, vocab_size = x.shape
        assert vocab_size == self.vocab_size, "Vocabulary size mismatch"
        x_emb = self.output_embedding(x)
        position_emb = self.positional_encoding(length, x_emb.shape[-1])
        output = x_emb + position_emb
        for _ in range(self.num_layers):
            output = self.decoder_block(output, encoder_output)
        return output