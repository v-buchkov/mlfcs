from __future__ import annotations

import torch
import torch.nn as nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class LSTMPredictor(AbstractPredictor):
    def __init__(
        self, hidden_size: int, n_features: int, n_layers: int, *args, **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        torch.manual_seed(12)

        self.model = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bias=False,
        )

        self.linear = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)

        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)  # Orthogonal initialization
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        model_device = past_returns.device

        past_returns = past_returns**2

        full_features = torch.cat([past_returns, features], dim=1)
        if hidden is None:
            h_t = torch.zeros(
                self.n_layers,
                self.hidden_size,
                dtype=torch.float32,
                requires_grad=True,
            ).to(model_device)
        else:
            h_t = hidden

        if memory is None:
            c_t = torch.zeros(
                self.n_layers,
                self.hidden_size,
                dtype=torch.float32,
                requires_grad=True,
            ).to(model_device)
        else:
            c_t = memory

        out, (h_t, c_t) = self.model(full_features, (h_t, c_t))
        out = self.layer_norm(out)
        out = self.linear(out)

        return nn.Softplus()(out), (h_t, c_t)
