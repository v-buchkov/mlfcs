from __future__ import annotations

import torch
import torch.nn as nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class LSTMPredictor(AbstractPredictor):
    def __init__(
        self, hidden_size: int, n_features: int, n_layers: int = 2, *args, **kwargs
    ):
        super().__init__()

        self.model = nn.LSTM(hidden_size, n_features, n_layers, *args, **kwargs)

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        full_features = torch.cat([past_returns, features], dim=1)
        return nn.Softplus()(self.model(full_features))
