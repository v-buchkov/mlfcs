from __future__ import annotations

import torch
import torch.nn as nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class LSTMPredictor(AbstractPredictor):
    def __init__(
        self, hidden_size: int, n_features: int, n_unique_features: int, n_layers: int, *args, **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_unique_features = n_unique_features
        self.n_layers = n_layers

        torch.manual_seed(12)

        self.model = nn.LSTM(
            input_size=n_unique_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bias=False,
        )

        self.final_layer = nn.Sequential(
            # nn.Linear(hidden_size * n_features // n_unique_features + 2, hidden_size),
            nn.Linear(hidden_size * n_layers * 2 + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)  # Orthogonal initialization
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization

    def _forward(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        model_device = past_returns.device

        sequence_length = features.shape[1] // self.n_unique_features

        # Reshape features to have shape (batches, sequence, features)
        features = features.reshape(
            features.shape[0], sequence_length, self.n_unique_features
        )

        if hidden is None:
            h_t = torch.zeros(
                self.n_layers,
                features.shape[0],
                self.hidden_size,
                dtype=torch.float32,
                requires_grad=True,
            ).to(model_device)
        else:
            h_t = hidden

        if memory is None:
            c_t = torch.zeros(
                self.n_layers,
                features.shape[0],
                self.hidden_size,
                dtype=torch.float32,
                requires_grad=True,
            ).to(model_device)
        else:
            c_t = memory

        out, (h_t, c_t) = self.model(features, (h_t, c_t))
        # out = out.reshape(features.shape[0], -1)
        # out = self.final_layer(torch.cat([out, past_returns, past_vols], dim=1))

        out = torch.cat([h_t.reshape(features.shape[0], -1), c_t.reshape(features.shape[0], -1)], dim=1)
        # print(out.shape)
        out = self.final_layer(torch.cat([out, past_returns, past_vols], dim=1))

        return nn.Softplus()(out), (h_t, c_t)
