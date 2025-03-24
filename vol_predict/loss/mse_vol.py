import torch
import torch.nn as nn

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss


class MSEVolLossAbstract(AbstractCustomLoss):
    def __init__(self):
        super().__init__()

    def forward(self, true_returns: torch.Tensor, pred_vol: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        true_vol = true_returns ** 2
        return nn.MSELoss()(true_vol, pred_vol)
