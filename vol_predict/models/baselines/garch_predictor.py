from __future__ import annotations

import torch

from vol_predict.models.abstract_predictor import AbstractPredictor


class GARCHPredictor(AbstractPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
