import copy
import torch
import torch.nn as nn

from .layers import SublayerWrapper, PositionalEncoding, MultiHeadAttention, FeedForwardLayer
from torchnmt.datasets.utils import Vocab


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, self_attn, ffn, dropout_r):
        super().__init__()
        self.self_attn = SublayerWrapper(dim, self_attn, dropout_r)
        self.ffn = SublayerWrapper(dim, ffn, dropout_r)

    def forward(self, x, src_mask):
        """
        Args:
            x: source input (bs, src_len, model_dim)
            src_mask: (bs, src_len, src_len)
        """
        x = self.self_attn(x, x, x, src_mask)
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers, heads, vocab_size, model_dim, ffn_dim, dropout_emb, dropout_r, **kwargs):
        super().__init__()
        c = copy.deepcopy
        self.embed = nn.Embedding(vocab_size + len(Vocab.extra), model_dim)
        self.pe = PositionalEncoding(model_dim)
        mha = MultiHeadAttention(heads, model_dim)
        ffn = FeedForwardLayer(model_dim, ffn_dim)
        layer = TransformerEncoderLayer(model_dim, c(mha), c(ffn), dropout_r)
        self.layers = nn.ModuleList([c(layer) for _ in range(layers)])

        self.dropout_emb = nn.Dropout(dropout_emb)

    def forward(self, x, src_mask):
        """
        Args:
            x: source input (bs, src_len, model_dim)
            src_mask: (bs, src_len, src_len)
        """
        x = self.embed(x)
        x = self.dropout_emb(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
