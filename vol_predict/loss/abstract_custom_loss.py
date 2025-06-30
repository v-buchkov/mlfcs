from __future__ import annotations

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import torch


import torch.nn as nn


class AbstractCustomLoss(nn.Module, ABC):
    def __init__(self, l2_coef: float = 0.0):
        super().__init__()
        self.l2_coef = l2_coef
    @abstractmethod
    def forward(
        self,
        true_returns: torch.Tensor,
        true_vols: torch.Tensor,
        pred_vol: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        true_returns: torch.Tensor,
        true_vols: torch.Tensor,
        pred_vol: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.forward(true_returns, true_vols, pred_vol, *args, **kwargs)

    def compute_l2(self, model: nn.Module) -> torch.Tensor:
        """
        Computes L2 penalty over model parameters. 
        l2_coef stands for the lambda parameter
        """
        if self.l2_coef == 0.0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        l2_sum = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        return self.l2_coef * l2_sum
