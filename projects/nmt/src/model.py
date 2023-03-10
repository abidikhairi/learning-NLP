import torch as th
import torch.nn as nn

from src.config import Seq2SeqConfig, OptimizerConfig
from src.parts import Embedding, PositionalEncoding, EncoderLayer, MultiHeadAttention, FeedForward, DecoderLayer
from src.utils import clones, learning_rate


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()

        self.linear = nn.Linear(d_model, vocab_size, False)

    def forward(self, x):
        x = self.linear(x)

        return x


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
                 dropout: float = 0.1,
                 pad_token: int = 0):
        super(Seq2Seq, self).__init__()

        self.pe = PositionalEncoding(d_model, dropout=dropout, max_len=10000)

        self.src_embedding = Embedding(src_vocab, d_model, padding_idx=pad_token)
        self.tgt_embedding = Embedding(tgt_vocab, d_model, padding_idx=pad_token)

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

    def encode(self, src, src_mask):
        src = self.pe(self.src_embedding(src))
        memory = self.encoder(src, src_mask)

        return memory

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.pe(self.tgt_embedding(tgt))

        output = self.decoder(tgt, memory, src_mask, tgt_mask)

        return output

    def validation_step(self, src, src_mask, tgt, tgt_mask):
        output = self(src, src_mask, tgt, tgt_mask)

        return self.generator(output)


def create_model_from_config(src_vocab: int, tgt_vocab: int, config: Seq2SeqConfig, **kwargs):
    device = kwargs.get('device', 'cpu')

    return Seq2Seq(src_vocab, tgt_vocab, config.d_model, config.nhead, config.num_decoder_layers,
                   config.num_encoder_layers, config.dim_feedforward, config.dropout, config.pad_token) \
        .to(device)


def load_model_from_config(src_vocab: int, tgt_vocab: int, config: Seq2SeqConfig, model_file: str, **kwargs):
    model: th.nn.Module = create_model_from_config(src_vocab, tgt_vocab, config, **kwargs)
    model.load_state_dict(th.load(model_file))
    model.eval()

    return model


def create_optimizer_from_config(config: OptimizerConfig, model: th.nn.Module):
    lr_wrapper_fn = lambda step: learning_rate(step=step, warmup_steps=config.warmup_steps, d_model=config.d_model)

    optimizer = th.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_wrapper_fn)

    return optimizer, scheduler
