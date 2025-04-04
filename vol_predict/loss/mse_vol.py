import torch
import torch.nn as nn

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss


class MSEVolLoss(AbstractCustomLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self, true_returns: torch.Tensor, true_vols: torch.Tensor, pred_vol: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return nn.MSELoss()(true_vols, pred_vol)
