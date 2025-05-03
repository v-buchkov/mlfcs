import torch

from vol_predict.loss.mse_vol import MSEVolLoss


class RMSEVolLoss(MSEVolLoss):
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
        return super().forward(true_returns, true_vols, pred_vol).sqrt()
