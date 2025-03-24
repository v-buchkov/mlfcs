from typing import Type
from dataclasses import dataclass

import torch
import torch.nn as nn

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss
from vol_predict.loss.loss import Loss
from vol_predict.loss.volatility_estimators import VolatilityMethod


@dataclass
class ModelConfig:
    LR: float = 1e-3
    HIDDEN_SIZE: int = 64

    TRAIN_EPOCHS: int = 100
    VAL_EPOCHS: int = 100

    OPTIMIZER: Type[torch.optim.Optimizer] = torch.optim.SGD

    WEIGHT_DECAY: float = 0.0

    BATCH_SIZE: int = 64

    LOSS: AbstractCustomLoss = Loss.MSE

    # TODO (V) handle metrics plotting
    METRICS: tuple[nn.Module] = (torch.nn.MSELoss,)

    VOL_CALC_METHOD: VolatilityMethod = VolatilityMethod("squared_returns")
