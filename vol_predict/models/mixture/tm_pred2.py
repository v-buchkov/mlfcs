from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import math
from abc import ABC
from vol_predict.models.abstract_predictor import AbstractPredictor

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractMixturePredictor(AbstractPredictor, ABC):
    """
    A general 2-component mixture predictor:
      - Gating that yields gate_weights[:,0] for AR, gate_weights[:,1] for Feature
      - Abstract methods for compute_ar_params(...) and compute_feat_params(...)
    Subclasses must also define how to interpret those params for a distribution.
    """

    def __init__(self, ar_order: int, n: int, lb: int):
        super().__init__()
        self.ar_order = ar_order
        self.n = n
        self.lb = lb

        # -- gating: AR gate is linear in past_returns
        self.ar_gate_lin = nn.Linear(ar_order, 1)

        # -- gating: Feature gate is bilinear in features
        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

    def bilinear_scalar(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        X: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        A in R^n
        B in R^lb
        X in R^[batch_size, n, lb]
        bias in R^[1]
        Return => shape [batch_size]
        """
        tmp = X * A.view(1, self.n, 1)  # shape [B, n, lb]
        tmp_sumF = tmp.sum(dim=1)      # shape [B, lb]
        tmp2 = tmp_sumF * B.view(1, self.lb)  # shape [B, lb]
        val = tmp2.sum(dim=1)               # shape [B]
        return val + bias

    def compute_gating_weights(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce gating scores [batch_size, 2], then softmax => [batch_size, 2].
        """
        # AR gating logit => shape [B,1]
        gate_ar_score = self.ar_gate_lin(past_returns)
        # Feature gating logit => shape [B,1]
        feat_gate_score = self.bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate
        ).unsqueeze(-1)

        gate_logits = torch.cat(
            [gate_ar_score, feat_gate_score], dim=1)  # [B,2]
        gate_weights = F.softmax(gate_logits, dim=1)  # [B,2]
        return gate_weights

    @abstractmethod
    def compute_ar_params(self, past_returns: torch.Tensor) -> dict:
        """
        Return distribution parameters for the AR component, e.g.
        {
          'mean': shape [B],
          'logvar': shape [B],
          ... or 'mu_logvol': shape [B], etc.
        }
        """
        pass

    @abstractmethod
    def compute_feat_params(self, features: torch.Tensor) -> dict:
        """
        Return distribution parameters for the Feature-based component.
        Same shape conventions as above.
        """
        pass

    @abstractmethod
    def component_mean(self, params: dict) -> torch.Tensor:
        """
        Return E[v] for that distribution, shape [B].
        """
        pass

    def forward(self, past_returns, features):
        """
        Return a dictionary with AR params, Feature params,
        gating weights, mixture mean, etc.
        So external code can compute the NLL or other losses.
        """
        # 1) compute distribution parameters for AR side
        ar_params = self.compute_ar_params(past_returns)
        # 2) compute distribution parameters for Feature side
        feat_params = self.compute_feat_params(features)
        # 3) gating
        gate_weights = self.compute_gating_weights(
            past_returns, features)  # [B,2]

        # 4) get each component's mean if you want to log or do MSE:
        ar_mean = self.component_mean(ar_params)        # [B]
        feat_mean = self.component_mean(feat_params)     # [B]
        mixture_mean = gate_weights[:, 0] * \
            ar_mean + gate_weights[:, 1]*feat_mean

        return {
            "ar_params": ar_params,
            "feat_params": feat_params,
            "gate_weights": gate_weights,    # shape [B,2]
            "ar_mean": ar_mean,             # shape [B]
            "feat_mean": feat_mean,         # shape [B]
            "mixture_mean": mixture_mean,   # shape [B]
        }


class TM_N_Predictor(AbstractMixturePredictor):
    """
    A temporal mixture model with Normal components.
    AR:      vol ~ Normal(ar_mean, ar_sigma^2)
    Feature: vol ~ Normal(feat_mean, feat_sigma^2)
    We'll store them as linear / bilinear transforms for mean + logvar.
    """

    def __init__(self, ar_order: int, n: int, lb: int):
        super().__init__(ar_order, n, lb)
        # AR
        self.ar_mean_lin = nn.Linear(ar_order, 1)    # for mean
        self.ar_logvar_lin = nn.Linear(ar_order, 1)  # for log-var

        # Feature side (bilinear)
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))

        self.A_logvar = nn.Parameter(torch.randn(n))
        self.B_logvar = nn.Parameter(torch.randn(lb))
        self.bias_logvar = nn.Parameter(torch.zeros(1))

    def compute_ar_params(self, past_returns: torch.Tensor) -> dict:
        mean = self.ar_mean_lin(past_returns).squeeze(-1)   # [B]
        logvar = self.ar_logvar_lin(past_returns).squeeze(-1)
        sigma = torch.exp(0.5*logvar)
        return {
            "mean": mean,   # shape [B]
            "sigma": sigma,  # shape [B]
        }

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        # bilinear helper
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            return tmp2.sum(dim=1) + bias

        mean = bilinear_scalar(self.A_mean, self.B_mean,
                               features, self.bias_mean)
        logvar = bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar)
        sigma = torch.exp(0.5*logvar)
        return {
            "mean": mean,
            "sigma": sigma
        }

    def component_mean(self, params: dict) -> torch.Tensor:
        # For Normal, E[X] = mean
        return params["mean"]


class TM_LN_Predictor(AbstractMixturePredictor):
    """
    A temporal mixture model with LogNormal components:
      AR side: log(vol) ~ Normal(ar_mean_logvol, ar_sigma^2)
      Feature side: log(vol) ~ Normal(feat_mean_logvol, feat_sigma^2).
    """

    def __init__(self, ar_order: int, n: int, lb: int):
        super().__init__(ar_order, n, lb)
        # AR side
        self.ar_mean_lin = nn.Linear(ar_order, 1)
        self.ar_logvar_lin = nn.Linear(ar_order, 1)

        # Feature side (bilinear)
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))

        self.A_logvar = nn.Parameter(torch.randn(n))
        self.B_logvar = nn.Parameter(torch.randn(lb))
        self.bias_logvar = nn.Parameter(torch.zeros(1))

    def compute_ar_params(self, past_returns):
        mean_logvol = self.ar_mean_lin(past_returns).squeeze(-1)
        logvar = self.ar_logvar_lin(past_returns).squeeze(-1)
        sigma = torch.exp(0.5*logvar)
        return {
            "mean_logvol": mean_logvol,
            "sigma": sigma
        }

    def compute_feat_params(self, features):
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            return tmp2.sum(dim=1) + bias

        mean_logvol = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)
        logvar = bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar)
        sigma = torch.exp(0.5*logvar)
        return {
            "mean_logvol": mean_logvol,
            "sigma": sigma
        }

    def component_mean(self, params: dict) -> torch.Tensor:
        # For lognormal with mu=params["mean_logvol"], sigma => E[X] = exp(mu + 0.5*sigma^2)
        mu = params["mean_logvol"]
        s = params["sigma"]
        return torch.exp(mu + 0.5*(s**2))


class TM_IG_Predictor(AbstractMixturePredictor):
    """
    Temporal mixture model with inverse Gaussian components.

    AR component: vol ~ InverseGaussian(mu_ar, lambda_ar)
    Feature component: vol ~ InverseGaussian(mu_feat, lambda_feat)

    For both components, the mean is simply mu.
    Here, lambda is modeled as a global trainable parameter.
    """

    def __init__(self, ar_order: int, n: int, lb: int):
        super().__init__(ar_order, n, lb)
        # AR: linear mapping for mu
        self.ar_mean_lin = nn.Linear(ar_order, 1)
        # Global lambda for AR component
        self.lambda_ar = nn.Parameter(torch.tensor(1.0))

        # Feature: bilinear for mu
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))
        # Global lambda for feature component
        self.lambda_feat = nn.Parameter(torch.tensor(1.0))

        # Gating parameters are inherited

    def compute_ar_params(self, past_returns: torch.Tensor) -> dict:
        mu = self.ar_mean_lin(past_returns).squeeze(-1)  # [B]
        # Expand global lambda to match batch shape
        lam = self.lambda_ar.expand_as(mu)
        return {"mu": mu, "lam": lam}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        mu = self.bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)  # [B]
        lam = self.lambda_feat.expand_as(mu)
        return {"mu": mu, "lam": lam}

    def component_mean(self, params: dict) -> torch.Tensor:
        # For inverse Gaussian, E[X] = mu.
        return params["mu"]


class TM_W_Predictor(AbstractMixturePredictor):
    """
    Temporal mixture model with Weibull components.

    AR component: v ~ Weibull(k_ar, lam_ar)
    Feature component: v ~ Weibull(k_feat, lam_feat)

    For the AR component, parameters are produced via a small MLP over past_returns.
    For the Feature component, parameters are produced via a small MLP over flattened features.

    The component mean for a Weibull distributed variable is:
      E[v] = lam * Gamma(1 + 1/k)
    """

    def __init__(self, ar_order: int, n: int, lb: int, eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # AR net: outputs raw estimates for k and lam
        self.ar_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # outputs raw_k and raw_lam
        )

        # Feature net: MLP over flattened features
        self.feat_net = nn.Sequential(
            nn.Linear(n * lb, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # outputs raw_k and raw_lam
        )

        # Gating: we already have self.ar_gate_lin and bilinear gate from the base class

    def compute_ar_params(self, past_returns: torch.Tensor) -> dict:
        raw = self.ar_net(past_returns)  # shape [B, 2]
        raw_k, raw_lam = raw.split(1, dim=-1)  # each shape [B,1]
        k = F.softplus(raw_k) + self.eps
        lam = F.softplus(raw_lam) + self.eps
        return {"k": k.squeeze(-1), "lam": lam.squeeze(-1)}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        batch_size = features.size(0)
        x_flat = features.view(batch_size, -1)  # flatten to [B, n*lb]
        raw = self.feat_net(x_flat)             # shape [B, 2]
        raw_k, raw_lam = raw.split(1, dim=-1)
        k = F.softplus(raw_k) + self.eps
        lam = F.softplus(raw_lam) + self.eps
        return {"k": k.squeeze(-1), "lam": lam.squeeze(-1)}

    def component_mean(self, params: dict) -> torch.Tensor:
        # For Weibull: mean = lam * Gamma(1 + 1/k)
        k = params["k"]
        lam = params["lam"]
        # Use torch.special.gammaln if available
        inv_k = (1.0 / k).clamp(max=10.0)
        gamma_val = torch.exp(torch.special.gammaln(1.0 + inv_k))
        return lam * gamma_val
