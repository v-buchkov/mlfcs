from enum import Enum

from vol_predict.loss.mse_vol import MSEVolLoss
from vol_predict.loss.rmse_vol import RMSEVolLoss
from vol_predict.loss.normal_nll import NormalNLL
from vol_predict.loss.bayesian_nll import BayesianNLL
from vol_predict.loss.tm_loss import MixtureNormalNLL, HingeNormalMixtureNLL, MixtureLogNormalNLL, MixtureInverseGaussianNLL, MixtureWeibullNLL


class Loss(Enum):
    MSE = MSEVolLoss
    RMSE = RMSEVolLoss
    NLL = NormalNLL
    BAYESIAN_NLL = BayesianNLL
    TM_N_NLL = lambda: MixtureNormalNLL(l2_coef=1e-4)
    TM_NH_NLL = HingeNormalMixtureNLL
    TM_LN_NLL = MixtureLogNormalNLL
    TM_IG_NLL = MixtureInverseGaussianNLL
    TM_W_NLL = MixtureWeibullNLL


class AvailableLosses(Enum):
    MSE = "mse"
    RMSE = "rmse"
    NLL = "nll"
    BAYESIAN_NLL = "bayesian_nll"
