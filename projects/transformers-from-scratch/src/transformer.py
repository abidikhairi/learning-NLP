import torch as th
import torch.nn as nn
from parts import MultiHeadAttention, FeedForwardNet


class EncoderLayer(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, nheads: int, d_model: int, d_ff: int = 1024) -> None:
        """Transformer encoder layer

        Args:
            in_feats (int): input dimension
            out_feats (int): hidden dimension
            nheads (int): number of attention heads
            d_model (int): model dimension
            d_ff (int, optional): feedforward dimension. Defaults to 1024.
        """
        super().__init__()

        self.d_k = out_feats // nheads
        self.nheads = nheads

        self.attention = MultiHeadAttention(in_feats=in_feats, out_feats=out_feats, nheads=nheads, d_model=d_model)
        self.ffn = FeedForwardNet(in_feats=d_model, out_feats=d_model, d_ff=d_ff)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: th.Tensor):
        x = self.attention(x)
        x = self.layer_norm(x)

        out = self.layer_norm(self.ffn(x) + x)

        return out


if __name__ == '__main__':
    x = th.rand(16, 10, 128)

    layer = EncoderLayer(128, 256, 8, 2048, 1024)

    out = layer(x)

    print(out.shape)
    print(layer)