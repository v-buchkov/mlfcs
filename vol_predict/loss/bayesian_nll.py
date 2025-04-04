import torch

from vol_predict.loss.normal_nll import NormalNLL


class BayesianNLL(NormalNLL):
    def __init__(self):
        super().__init__()

    def forward(
        self, true_returns: torch.Tensor, true_vols: torch.Tensor, pred_vol: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor:
        return (
            super().forward(true_returns, true_vols, pred_vol)
            + ((pred_vol - prior) ** 2).sum(dim=1).mean()
        )
