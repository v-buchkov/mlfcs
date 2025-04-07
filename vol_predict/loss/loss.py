from enum import Enum

from vol_predict.loss.mse_vol import MSEVolLoss
from vol_predict.loss.rmse_vol import RMSEVolLoss
from vol_predict.loss.normal_nll import NormalNLL
from vol_predict.loss.bayesian_nll import BayesianNLL


class Loss(Enum):
    MSE = MSEVolLoss
    RMSE = RMSEVolLoss
    NLL = NormalNLL
    BAYESIAN_NLL = BayesianNLL


class AvailableLosses(Enum):
    MSE = "mse"
    RMSE = "rmse"
    NLL = "nll"
    BAYESIAN_NLL = "bayesian_nll"
