import torch as th
import torch.nn as nn


class AttentionHead(nn.Module):
    
    def __init__(self, in_feats: int, d_k: int) -> None:
        """Single attention head from \"Attention Is All You Need\"  

        Args:
            in_feats (int): input features dim
            d_k (int): input features dim
        """
        super().__init__()

        self.w_q = nn.Linear(in_features=in_feats, out_features=d_k, bias=False)
        self.w_k = nn.Linear(in_features=in_feats, out_features=d_k, bias=False)
        self.w_v = nn.Linear(in_features=in_feats, out_features=d_k, bias=False)

        self.d_k = d_k


    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor, return_attn=False):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        unnormalized_scores = th.bmm(q, k.permute(0, 2, 1))
        scaled_scores = unnormalized_scores * (1 / (self.d_k ** 0.5))
        normalized_scores = th.softmax(scaled_scores, dim=-1)

        out = th.bmm(normalized_scores, v)
        
        if return_attn:
            return out, normalized_scores
        
        return out


class MultiHeadAttention(nn.Module):


    def __init__(self, in_feats: int, out_feats: int, nheads: int, d_model: int) -> None:
        """Applies parallel attention layers to the input sequence

        Args:
            in_feats (int): input features dim
            out_feats (int): linear layer input dim
            nheads (int): number of attention heads
            d_model (int): attention output dim
        """
        super().__init__()

        assert out_feats % nheads == 0, "out_feats must be divisible by nheads, got {} and {}".format(out_feats, nheads)

        self.d_k = out_feats // nheads
        self.nheads = nheads

        heads = [AttentionHead(in_feats=in_feats, d_k=self.d_k) for _ in range(nheads)]

        self.heads = nn.ModuleList(heads)

        self.linear = nn.Linear(in_features=self.d_k * self.nheads, out_features=d_model, bias=False)


    def forward(self, x: th.Tensor):
        out = [self.heads[idx](x, x, x) for idx in range(self.nheads)]

        feats = th.cat(out, dim=-1)

        return self.linear(feats)


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, d_ff: int = 1024) -> None:
        """Applies two layer perceptron to input

        Args:
            in_feats (int): input dimension
            out_feats (int): output dimension
            d_ff (int, optional): hidden dimension. Defaults to 1024.
        """
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=d_ff),
            nn.ReLU(),
            nn.Linear(in_features=d_ff, out_features=out_feats)
        )

    def forward(self, x: th.Tensor):
        return self.ffn(x)


if __name__ == "__main__":
    x = th.randn(16, 10, 128)

    attention = MultiHeadAttention(in_feats=128, out_feats=256, nheads=8, d_model=2048)
    ffn = FeedForwardNet(2048, 2048)

    out = attention(x)
    y = ffn(out)

    print(out.shape)
    print(y.shape)
