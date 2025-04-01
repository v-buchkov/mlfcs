from typing import Type
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss
from vol_predict.loss.loss import Loss
from vol_predict.loss.volatility_estimators import VolatilityMethod


@dataclass
class ModelConfig:
    lr: float = 1e-3
    hidden_size: int = 64
    n_layers: int = 2

    n_epochs: int = 100
    n_features: int = 24

    optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD

    weights_decay: float = 0.0

    batch_size: int = 64

    # EWMA hyperparameters
    ewma_look_back_win: int = 21
    ewma_vola_calc_win: int = 21

    loss: AbstractCustomLoss = Loss.MSE

    # TODO (V) handle metrics plotting
    metrics: tuple[nn.Module] = (torch.nn.MSELoss,)

    vol_calc_method: VolatilityMethod = VolatilityMethod("squared_returns")

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
