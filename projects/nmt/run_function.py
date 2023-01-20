import sys
import torch as th
import matplotlib.pyplot as plt
from src.parts import PositionalEncoding
from src.utils import subsequent_mask


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


if __name__ == "__main__":
    function_name = sys.argv[1]
    getattr(sys.modules[__name__], function_name)()
