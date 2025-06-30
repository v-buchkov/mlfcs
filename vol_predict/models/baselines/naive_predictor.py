from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from vol_predict.models.abstract_predictor import AbstractPredictor


class NaivePredictor(AbstractPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.params = nn.Parameter(requires_grad=True)
        self._trailing_vars = []

    def _forward(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        vol_est = np.mean(self._trailing_vars) if len(self._trailing_vars) > 0 else 0
        pred_vols = torch.ones_like(past_returns) * vol_est
        self._trailing_vars += past_vols.tolist()
        return pred_vols
