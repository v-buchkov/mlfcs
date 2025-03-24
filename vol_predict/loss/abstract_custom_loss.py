from __future__ import annotations

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import torch


import torch.nn as nn


class AbstractCustomLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, true_returns: torch.Tensor, pred_vol: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, true_returns: torch.Tensor, pred_vol: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward(true_returns, pred_vol, *args, **kwargs)
