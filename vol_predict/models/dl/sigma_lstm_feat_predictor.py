from __future__ import annotations

import torch
import torch.nn as nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class SigmaLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        self.weight_o = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, input, state):
        hx, cx = state
        gates = (
            input @ self.weight_ih.T
            + self.bias_ih
            + hx @ self.weight_hh.T
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)

        out_vol = torch.mm(torch.pow(cy, 2), self.weight_o.t())
        out = torch.normal(mean=0, std=out_vol)

        hy = out * torch.tanh(cy)

        return hy, cy


class SigmaLSTMFeatPredictor(AbstractPredictor):
    def __init__(
        self,
        hidden_size: int,
        n_features: int,
        n_unique_features: int,
        n_layers: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_unique_features = n_unique_features
        self.n_layers = n_layers

        torch.manual_seed(12)

        self.lstm_cells = nn.Sequential(
            *[
                SigmaLSTMCell(n_unique_features, hidden_size)
                for _ in range(self.n_layers)
            ]
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
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model_device = past_returns.device

        features = features[:, 12:]

        sequence_length = features.shape[1] // self.n_unique_features

        # Reshape features to have shape (batches, sequence, features)
        features = features.reshape(
            features.shape[0], sequence_length, self.n_unique_features
        )

        h_t = torch.zeros(
            features.shape[0],
            self.hidden_size,
            dtype=torch.float32,
            requires_grad=True,
        ).to(model_device)

        c_t = torch.zeros(
            features.shape[0],
            self.hidden_size,
            dtype=torch.float32,
            requires_grad=True,
        ).to(model_device)

        for cell in self.lstm_cells:
            for i in range(sequence_length):
                h_t, c_t = cell(features[:, i, :], (h_t, c_t))

        return (c_t.mean(dim=1).unsqueeze(1)) ** 2
