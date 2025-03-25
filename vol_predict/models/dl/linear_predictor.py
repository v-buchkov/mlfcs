from __future__ import annotations

import torch
from torch import nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class MLPPredictor(AbstractPredictor):
    def __init__(self, n_features: int, *args, **kwargs):
        super().__init__()

        self.model = nn.Linear(n_features, 1)

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        full_features = torch.cat([past_returns, features], dim=1)
        return self.model(full_features)
