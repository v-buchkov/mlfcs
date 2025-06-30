from __future__ import annotations

from abc import ABC, abstractmethod

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
        past_vols: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(past_returns, past_vols, features, *args, **kwargs)

    def __call__(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.forward(past_returns, past_vols, features, *args, **kwargs)

    @abstractmethod
    def _forward(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
