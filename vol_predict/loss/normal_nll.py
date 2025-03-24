import torch

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss


class NormalNLL(AbstractCustomLoss):
    def __init__(self):
        super().__init__()

    def forward(self, true_returns: torch.Tensor, pred_vol: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
