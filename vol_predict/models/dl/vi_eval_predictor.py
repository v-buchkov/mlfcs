from __future__ import annotations

import torch

from vol_predict.models.abstract_predictor import AbstractPredictor
from vol_predict.models.dl.lstm_vi_softplus_predictor import LSTMViSoftplusPredictor
from vol_predict.models.baselines.naive_predictor import NaivePredictor


class ViEvalPredictor(AbstractPredictor):
    def __init__(
        self,
        hidden_size: int,
        n_features: int,
        n_unique_features: int,
        n_layers: int,
        n_experiments: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_unique_features = n_unique_features
        self.n_layers = n_layers
        self.n_experiments = n_experiments

        torch.manual_seed(12)

        self.lstm = LSTMViSoftplusPredictor(
            hidden_size=hidden_size,
            n_features=n_features,
            n_unique_features=n_unique_features,
            n_layers=n_layers,
        )

        self.naive = NaivePredictor()

        self.seen_uncerts = []
        # self.c = 0

    def _forward(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.training:
            self.naive(past_returns, past_vols, features)
            return self.lstm(past_returns, past_vols, features)

        out = []
        for _ in range(self.n_experiments):
            out.append(self.lstm(past_returns, past_vols, features))
        out = torch.stack(out)

        lstm_pred = out.mean(dim=0)
        uncert = out.std(dim=0)

        self.seen_uncerts.append(uncert.mean().item())

        uncert_scaled = (uncert - min(self.seen_uncerts)) / (
            max(self.seen_uncerts) - min(self.seen_uncerts) + 1e-6
        )
        weight_lstm = 1 - torch.clamp(uncert_scaled, 0, 1) + 1e-6

        pred = weight_lstm * lstm_pred + (1 - weight_lstm) * self.naive(
            past_returns, past_vols, features
        )

        return pred
