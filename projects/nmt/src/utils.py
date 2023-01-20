import torch
import copy


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (size, size)

    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return mask == 0


def clones(module: torch.nn.Module, n: int = 2):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
