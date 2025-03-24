from enum import Enum

from vol_predict.loss.mse_vol import MSEVolLossAbstract
from vol_predict.loss.normal_nll import NormalNLL
from vol_predict.loss.bayesian_nll import BayesianNLL


class Loss(Enum):
    MSE = MSEVolLossAbstract
    NLL = NormalNLL
    BAYESIAN_NLL = BayesianNLL

class AvailableLosses(Enum):
    MSE = "mse"
    NLL = "nll"
    BAYESIAN_NLL = "bayesian_nll"
