from typing import Type
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss
from vol_predict.loss.loss import Loss
from vol_predict.loss.volatility_estimators import VolatilityMethod


@dataclass
class ModelConfig:
    lr: float = 1e-3
    hidden_size: int = 64
    n_layers: int = 2
    dropout: float = 0.0

    n_epochs: int = 100
    n_features: int | None = None
    n_unique_features: int | None = None

    optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD

    weights_decay: float = 0.0

    batch_size: int = 64

    loss: AbstractCustomLoss = Loss.MSE

    scaler: Type[BaseEstimator] = MinMaxScaler()

    # TODO (V) handle metrics plotting
    metrics: tuple[nn.Module] = (Loss.RMSE,)

    vol_calc_method: VolatilityMethod = VolatilityMethod("squared_returns")

    # Transformer
    n_attention_heads: int = 8 * 11
    dim_feedforward: int = 2048

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
