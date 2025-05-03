import torch

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss


class NormalNLL(AbstractCustomLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        true_returns: torch.Tensor,
        true_vols: torch.Tensor,
        pred_vol: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return (
            torch.log(pred_vol + 1e-12) + true_returns**2 / (pred_vol + 1e-12)
        ).sum()
