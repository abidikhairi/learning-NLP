import torch as th
import torch.nn as nn

from src.parts import Embedding, PositionalEncoding, EncoderLayer, MultiHeadAttention, FeedForward, DecoderLayer
from src.utils import clones


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()

        self.linear = nn.Linear(d_model, vocab_size, False)

    def forward(self, x):
        x = self.linear(x)

        return th.log_softmax(x, dim=-1)


class Encoder(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, feedforward: nn.Module, dropout: float, n: int = 2):
        super(Encoder, self).__init__()

        layer = EncoderLayer(d_model, self_attn, feedforward, dropout)
        self.layers = clones(layer, n)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, feedforward: nn.Module, dropout: float, n: int = 2):
        super(Decoder, self).__init__()

        layer = DecoderLayer(d_model, self_attn, feedforward, dropout)
        self.layers = clones(layer, n)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return x


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 512, nhead: int = 8, num_decoder_layers: int = 6,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super(Seq2Seq, self).__init__()

        self.pe = PositionalEncoding(d_model, dropout=dropout, max_len=10000)

        self.src_embedding = Embedding(src_vocab, d_model)
        self.tgt_embedding = Embedding(tgt_vocab, d_model)

        self_attn: nn.Module = MultiHeadAttention(d_model=d_model, nhead=nhead)
        feedforward: nn.Module = FeedForward(d_model=d_model, dim_ff=dim_feedforward)

        self.encoder = Encoder(d_model, self_attn=self_attn, feedforward=feedforward, dropout=dropout,
                               n=num_encoder_layers)
        self.decoder = Decoder(d_model, self_attn=self_attn, feedforward=feedforward, dropout=dropout,
                               n=num_decoder_layers)

        self.generator = Generator(d_model, vocab_size=tgt_vocab)

    def forward(self, src, src_mask, tgt, tgt_mask):
        src = self.pe(self.src_embedding(src))
        tgt = self.pe(self.tgt_embedding(tgt))

        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, src_mask, tgt_mask)

        return out
