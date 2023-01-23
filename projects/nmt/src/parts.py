import math
import torch as th
import torch.nn as nn
from src.utils import clones, subsequent_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, do_scale=True, padding_idx=0):
        super(Embedding, self).__init__()

        self.lookup = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        self.scale = do_scale

    def forward(self, x):
        if not self.scale:
            return self.lookup(x)
        x = self.lookup(x) * (self.d_model ** 0.5)
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_k: int, d_model: int):
        super(AttentionHead, self).__init__()

        self.w_q = nn.Linear(d_model, d_k, False)
        self.w_k = nn.Linear(d_model, d_k, False)
        self.w_v = nn.Linear(d_model, d_k, False)

        self.d_model = d_model

    def forward(self, query: th.Tensor, key: th.Tensor, value: th.Tensor, mask=None):
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        scores = th.bmm(query, key.transpose(1, 2)) * (self.d_model ** -0.5)
        # this also can be used -> key.permute(0, 2, 1)

        if mask is not None:
            scores = scores.masked_fill(mask, 1e-9)
        scores = th.softmax(scores, dim=-1)

        attns = th.bmm(scores, value)

        return attns


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super(MultiHeadAttention, self).__init__()

        assert d_model // nhead
        self.d_k = d_model // nhead

        self.heads = clones(AttentionHead(d_k=self.d_k, d_model=d_model), n=nhead)
        self.linear = nn.Linear(d_model, d_model, False)

    def forward(self, query: th.Tensor, key: th.Tensor, value: th.Tensor, mask: th.Tensor = None):
        heads = [head(query, key, value, mask) for head in self.heads]
        x = th.cat(heads, dim=-1)

        return self.linear(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int = 2048):
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, dim_ff)
        self.w2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        x = th.relu(self.w1(x))

        return self.w2(x)


class LayerApplier(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(LayerApplier, self).__init__()

        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer: callable):
        x = layer(x) + x
        return self.dropout(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, feedforward: nn.Module, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feedforward = feedforward
        self.sublayers = clones(LayerApplier(size=d_model, dropout=dropout), n=2)

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))

        return self.sublayers[1](x, lambda x: self.feedforward(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, feedforward: nn.Module, dropout: float):
        super(DecoderLayer, self).__init__()

        self.self_attn = self_attn
        self.src_attn = self_attn
        self.feedforward = feedforward

        self.sublayers = clones(LayerApplier(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(memory, memory, x, src_mask))

        return self.sublayers[2](x, lambda x: self.feedforward(x))


if __name__ == '__main__':
    memory = th.randn(16, 10, 512)
    tgt = th.randn(16, 10, 512)

    tgt_mask = subsequent_mask(10)
    src_mask = th.ones(16, 10, 10)

    self_attn = MultiHeadAttention(d_model=512, nhead=8)
    feedforward = FeedForward(d_model=512, dim_ff=2048)

    layer = DecoderLayer(d_model=512, self_attn=self_attn, feedforward=feedforward, dropout=0.1)

    out = layer(tgt, memory, src_mask, tgt_mask)

    print(out.shape)
