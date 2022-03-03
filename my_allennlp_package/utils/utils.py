import torch
from typing import Optional


def batched_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    if b is not None:
        y = y + b.unsqueeze(1)
    return y
