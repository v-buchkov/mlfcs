from __future__ import annotations

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class AbstractPredictor(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.hidden = None
        self.memory = None

    def forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward(past_returns, features)

    def __call__(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(past_returns, features)

    @abstractmethod
    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
