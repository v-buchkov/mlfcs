from __future__ import annotations

import torch
from torch import nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPPredictor(AbstractPredictor):
    def __init__(
        self, hidden_size: int, n_features: int, n_layers: int = 2, *args, **kwargs
    ):
        super().__init__()

        self.model = MLP([n_features] + [hidden_size] * n_layers + [1])

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        full_features = torch.cat([past_returns, features], dim=1)
        return self.model(full_features)
