import sys
import torch as th
import matplotlib.pyplot as plt

from src.model import Seq2Seq
from src.parts import PositionalEncoding
from src.utils import subsequent_mask, learning_rate


def positional_encoding():
    max_position = 256
    d_model=16
    pe = PositionalEncoding(d_model=d_model, dropout=0.0)
    x = th.zeros(max_position, d_model)

    out = pe(x)
    out = out[1].numpy()

    plt.plot()
    plt.imshow(out, aspect='auto', interpolation='nearest')
    plt.xlabel("Position")
    plt.ylabel("Encoding")
    plt.colorbar()
    plt.savefig("figures/positional_encoding.png")


def decoder_mask():
    mask = subsequent_mask(20).numpy()

    plt.plot()
    plt.imshow(mask)
    plt.xlabel("Available Context")
    plt.ylabel("Word")
    plt.colorbar()
    plt.savefig("figures/decoder_mask.png")


def learning_rate_scheduler():
    src_vocab = 10000
    tgt_vocab = 10000
    d_model = 1536
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    warmup_steps = 4000
    lr_wrapper_fn = lambda step: learning_rate(step=step, warmup_steps=warmup_steps, d_model=d_model)

    model = Seq2Seq(src_vocab, tgt_vocab, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

    optimizer = th.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_wrapper_fn)

    rates = []

    for step in range(20000):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    plt.figure(figsize=(14, 8))
    plt.plot(range(20000), rates)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.savefig("figures/lr_scheduler.png", transparent=False)


if __name__ == "__main__":
    function_name = sys.argv[1]
    getattr(sys.modules[__name__], function_name)()
