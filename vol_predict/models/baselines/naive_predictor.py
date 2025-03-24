from __future__ import annotations

import torch

from vol_predict.models.abstract_predictor import AbstractPredictor


class NaivePredictor(AbstractPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self._trailing_var = []

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones_like(past_returns) * past_returns.var(dim=0).item()
