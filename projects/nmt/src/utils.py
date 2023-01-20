import torch


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (size, size)

    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
