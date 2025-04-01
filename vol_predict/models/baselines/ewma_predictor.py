from __future__ import annotations

import torch
from torch.nn import Parameter
from torch import nn

from vol_predict.models.abstract_predictor import AbstractPredictor


class EWMAPredictor(AbstractPredictor, nn.Module):
    """
    Exponentially Weighted Moving Average (EWMA) volatility predictor. Given the past returns,
    it computes the volatility using `ewma_vola_calc_win` returns per estimation
    and averages them over the past `ewma_look_back_win` estimations of volatility.
    """

    def __init__(
        self,
        ewma_look_back_win: int,  # number of points for exp. averaging
        ewma_vola_calc_win: int,  # number of points for vola estimation
        *args,
        **kwargs,
    ):
        super().__init__()

        assert ewma_look_back_win > 0.0, "look_back_period must be greater than 0"
        assert ewma_vola_calc_win > 0.0, "vola_calc_window must be greater than 0"
        self.look_back = ewma_look_back_win
        self.vola_calc_win = ewma_vola_calc_win

        # sigmoid_inverse(3.4) = 0.9677, which is a forgetting_factor equivalent to half-life of 21
        self.raw_forgetting_factor = Parameter(torch.Tensor([3.4]), requires_grad=True)
        self.forgetting_factor = torch.sigmoid(self.raw_forgetting_factor)

    def _hl_to_ff(self, half_life: torch.Tensor) -> torch.Tensor:
        """
        Convert half-life to forgetting factor.
        """
        assert half_life.item() > 0.0, "half_life must be greater than 0"
        return torch.Tensor(0.5 ** (1 / half_life))

    def _ff_to_hl(self, forgetting_factor: torch.Tensor) -> torch.Tensor:
        """
        Convert forgetting factor to half-life.
        """
        assert forgetting_factor.item() > 0.0, (
            "forgetting_factor must be greater than 0"
        )
        assert forgetting_factor.item() < 1.0, "forgetting_factor must be less than 1"
        return -torch.log(torch.tensor(0.5)) / torch.log(
            torch.tensor(forgetting_factor)
        )

    def get_half_life(self) -> float:
        """
        Get the half-life of the forgetting factor.
        """
        return self._ff_to_hl(self.forgetting_factor.item())

    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:

        assert len(past_returns) >= self.look_back, (
            f"too few past returns: ewma_look_back_win={self.look_back}, \
                but {len(past_returns)} returns are provided"
        )

        # estimate the volatility using the past returns `self.vola_calc_win` returns
        vola_estimates = torch.zeros((self.look_back))
        for i in range(self.look_back):
            vola_estimates[i] = past_returns[-i - 1 - self.vola_calc_win : -i - 1].var()

        # predict the volatility as EWMA of the past volatility estimates
        normlaization_factor = (1 - self.forgetting_factor) / (
            1 - self.forgetting_factor**self.look_back
        )

        exp_weights = torch.zeros((self.look_back))
        for i in range(self.look_back):
            exp_weights[i] = self.forgetting_factor ** (i)

        return normlaization_factor * torch.sum(exp_weights * vola_estimates)
