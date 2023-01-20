import torch as th
import torch.nn as nn

from src.parts import Embedding, PositionalEncoding


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()

        self.linear = nn.Linear(d_model, vocab_size, False)

    def forward(self, x):
        x = self.linear(x)

        return th.log_softmax(x, dim=-1)


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model=512, nhead=8, num_decoder_layers=6, num_encoder_layers=6,
                 dim_feedforward=2048):
        super(Seq2Seq, self).__init__()

        self.pe = PositionalEncoding(d_model, dropout=0.1, max_len=10000)

        self.src_embedding = Embedding(src_vocab, d_model)
        self.tgt_embedding = Embedding(tgt_vocab, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_decoder_layers=num_decoder_layers,
                                          num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
                                          batch_first=True)

        self.generator = Generator(d_model, tgt_vocab)

    def forward(self, src, src_mask, tgt, tgt_mask):
        src_embed = self.pe(self.src_embedding(src))
        tgt_embed = self.pe(self.tgt_embedding(tgt))

        out = self.transformer(src_embed, tgt_embed, src_mask=src_mask, tgt_mask=tgt_mask)

        return self.generator(out)
