from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractPredictor(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.hidden = None
        self.memory = None

        self._dummy_param = nn.Parameter(torch.empty(0))

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
