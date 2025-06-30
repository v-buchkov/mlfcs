from __future__ import annotations

import torch
import torch.nn as nn

from vol_predict.models.dl.mlp import MLP
from vol_predict.models.abstract_predictor import AbstractPredictor


class MLPPredictor(AbstractPredictor):
    def __init__(
        self, hidden_size: int, n_features: int, n_layers: int = 2, *args, **kwargs
    ):
        super().__init__()

        self.model = MLP([1212] + ([hidden_size] * n_layers) + [1])

    def _forward(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        vol = self.model(features)
        return nn.Softplus()(vol)
