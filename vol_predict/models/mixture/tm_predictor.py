import torch

from vol_predict.models.abstract_predictor import AbstractPredictor


class TMPredictor(AbstractPredictor):
    def __init__(self):
        super().__init__()

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
