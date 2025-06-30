from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import math
from abc import ABC
from vol_predict.models.abstract_predictor import AbstractPredictor
from torch.distributions import LogNormal, Weibull, Normal
from vol_predict.models.distributions.inverse_gaussian import InverseGaussian

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

        # -- gating: AR gate is linear in past_volatility
        self.ar_gate_lin = nn.Linear(ar_order, 1)

        # -- gating: Feature gate is bilinear in features
        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

        # -- temperature gating

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.ar_param_key = "mean"
        self.feat_param_key = "mean"
    # def bilinear_scalar(
    #     self,
    #     A: torch.Tensor,
    #     B: torch.Tensor,
    #     X: torch.Tensor,
    #     bias: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     A in R^n
    #     B in R^lb
    #     X in R^[batch_size, n, lb]
    #     bias in R^[1]
    #     Return => shape [batch_size]
    #     """
    #     tmp = X * A.view(1, self.n, 1)  # shape [B, n, lb]
    #     tmp_sumF = tmp.sum(dim=1)      # shape [B, lb]
    #     tmp2 = tmp_sumF * B.view(1, self.lb)  # shape [B, lb]
    #     val = tmp2.sum(dim=1)               # shape [B]
    #     return val + bias

    def bilinear_scalar(self, A, B, X, bias):
        tmp = X * A.view(1, self.n, 1)
        tmp = tmp.sum(dim=1)
        tmp = tmp * B.view(1, self.lb)
        return tmp.sum(dim=1) / math.sqrt(self.lb) + bias

    def compute_gating_weights(
        self,
        past_volatility: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce gating scores [batch_size, 2], then softmax => [batch_size, 2].
        """
        # AR gating logit => shape [B,1]
        gate_ar_score = self.ar_gate_lin(past_volatility)
        # Feature gating logit => shape [B,1]
        feat_gate_score = self.bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate
        ).unsqueeze(-1)

        gate_logits = torch.cat(
            [gate_ar_score, feat_gate_score], dim=1)  # [B,2]
        gate_logits = gate_logits / self.temperature.clamp(0.1, 10.0)
        gate_weights = F.softmax(gate_logits, dim=1)  # [B,2]
        return gate_weights

    def get_prior_distribution(self, dist_type: str, params: dict,
                               roll_mean: torch.Tensor = None, param_key: str = "mean") -> torch.distributions.Distribution:
        """
        Generates a prior distribution for the given component parameters.
        dist_type: one of ["normal", "lognormal", "weibull", "invgauss"]
        params   : dict with keys like "mean", "sigma", etc.
        roll_mean: optional [B] prior center for mu
        """
        B = params[param_key].shape[0]
        device = params[param_key].device
        if roll_mean is None:
            roll_mean = torch.zeros(B, device=device)

        if dist_type == "normal":
            prior_mu = roll_mean
            prior_sigma = torch.ones_like(roll_mean)
            return torch.distributions.Normal(loc=prior_mu, scale=prior_sigma)

        elif dist_type == "lognormal":
            loc = roll_mean
            scale = torch.ones_like(roll_mean)
            return torch.distributions.LogNormal(loc=loc, scale=scale)

        elif dist_type == "weibull":
            shape = torch.distributions.Gamma(1.5, 1.0).sample((B,)).to(device)
            scale = torch.distributions.LogNormal(
                0.0, 0.5).sample((B,)).to(device)
            return torch.distributions.Weibull(scale=scale, concentration=shape)

        elif dist_type == "invgauss":
            mean = torch.exp(torch.normal(0.0, 0.5, size=(B,), device=device))
            lam = torch.distributions.Gamma(2.0, 1.0).sample((B,)).to(device)
            return InverseGaussian(loc=mean, concentration=lam)

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    @abstractmethod
    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
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

    @abstractmethod
    def get_distribution(self, params: dict, component: str):
        """
        Zwraca odpowiedni torch.distributions.* dla podanych parametrów.
        """
        pass

    def forward(self, past_volatility, features, roll_mean=None):
        """
        Return a dictionary with AR params, Feature params,
        gating weights, mixture mean, etc.
        So external code can compute the NLL or other losses.
        """
        batch_size = features.shape[0]
        total_features = self.n * self.lb
        assert features.shape[1] == total_features, (
            f"Expected flat feature shape [B, {total_features}], got {features.shape}"
        )

        features_reshaped = features.view(batch_size, self.n, self.lb)

        # 1) compute distribution parameters for AR side
        ar_params = self.compute_ar_params(past_volatility)
        # 2) compute distribution parameters for Feature side
        feat_params = self.compute_feat_params(features_reshaped)
        # 3) gating
        gate_weights = self.compute_gating_weights(
            past_volatility, features_reshaped)  # [B,2]

        # 4) get each component's mean if you want to log or do MSE:
        ar_mean = self.component_mean(ar_params)        # [B]
        feat_mean = self.component_mean(feat_params)     # [B]
        mixture_mean = gate_weights[:, 0] * \
            ar_mean + gate_weights[:, 1]*feat_mean

        ar_dist = self.get_distribution(ar_params,   component="ar")
        feat_dist = self.get_distribution(feat_params, component="feat")

        prior_ar = self.get_prior_distribution(
            self.ar_prior_type, ar_params, roll_mean, param_key=self.ar_param_key)
        prior_feat = self.get_prior_distribution(
            self.feat_prior_type, feat_params, roll_mean, param_key=self.feat_param_key)

        return {
            "ar_params": ar_params,
            "feat_params": feat_params,
            "gate_weights": gate_weights,    # shape [B,2]
            "ar_mean": ar_mean,             # shape [B]
            "feat_mean": feat_mean,         # shape [B]
            "mixture_mean": mixture_mean,   # shape [B]
            "dists": [ar_dist, feat_dist],
            "prior_dists": [prior_ar, prior_feat]
        }

    def _forward(self, past_volatility, features, *args, **kwargs):
        return self.forward(past_volatility, features)


class TM_N_Predictor(AbstractMixturePredictor):
    """
    Temporal Mixture – Gaussian components (TM-G).
    AR-side  :  vol ~ N(mu_ar,  sigma_ar^2)
    Feat-side:  vol ~ N(mu_feat, sigma_feat^2)

    We return a dictionary:
        {"mean": mu, "sigma": sigma}
    """

    def __init__(self, ar_order: int, n: int, lb: int):
        super().__init__(ar_order, n, lb)

        # ---------- AR component ----------
        self.ar_mean_lin = nn.Linear(ar_order, 1)   # mu
        self.ar_logvar_lin = nn.Linear(ar_order, 1)   # log(sigma^2)

        # ---------- Feature component (bilinear) ----------
        # Mean
        self.A_mean = nn.Parameter(0.001 * torch.randn(n))
        self.B_mean = nn.Parameter(0.001 * torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))

        # Log-variance
        self.A_logvar = nn.Parameter(0.001 * torch.randn(n))
        self.B_logvar = nn.Parameter(0.001 * torch.randn(lb))
        self.bias_logvar = nn.Parameter(torch.zeros(1))

        self.ar_prior_type = "normal"
        self.feat_prior_type = "normal"
        self.ar_param_key = "mean"
        self.feat_param_key = "mean"
    # ---- helping ----
    # def bilinear_scalar(self, A: torch.Tensor, B: torch.Tensor,
    #                     X: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    #     """
    #     X : [B, n, lb]
    #     A : [n]   (features)
    #     B : [lb]  (time)
    #     Zwrot: [B]
    #     """
    #     tmp = X * A.view(1, self.n, 1)     # broadcast po osi feature
    #     tmp = tmp.sum(dim=1)               # zsumuj po feature → [B, lb]
    #     tmp = tmp * B.view(1, self.lb)     # waga czasowa
    #     return tmp.sum(dim=1) + bias       # → [B]
    # def bilinear_scalar(self, A, B, X, bias):
    #     tmp = X * A.view(1, self.n, 1)
    #     tmp = tmp.sum(dim=1)
    #     tmp = tmp * B.view(1, self.lb)
    #     return tmp.sum(dim=1) / math.sqrt(self.lb) + bias

    # ---- AR component ----
    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        mu = self.ar_mean_lin(past_volatility).squeeze(-1)          # [B]
        logvar = self.ar_logvar_lin(past_volatility).squeeze(-1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        sigma = torch.exp(0.5 * logvar).clamp(min=1e-3, max=1e2)       # [B]
        return {"mean": mu, "sigma": sigma}

    # ---- Feature component ----
    def compute_feat_params(self, features: torch.Tensor) -> dict:
        mu = self.bilinear_scalar(self.A_mean, self.B_mean,
                                  features, self.bias_mean)

        logvar = self.bilinear_scalar(self.A_logvar, self.B_logvar,
                                      features, self.bias_logvar)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        sigma = torch.exp(0.5 * logvar).clamp(min=1e-3, max=1e2)
        return {"mean": mu, "sigma": sigma}

    # ---- For loss/eval ----
    def component_mean(self, params: dict) -> torch.Tensor:
        # For Gauss: E[X] = mean
        return params["mean"]

    def component_mean(self, params: dict) -> torch.Tensor:
        # For Normal, E[X] = mean
        return params["mean"]

    def get_distribution(self, params: dict, component: str = "ar"):
        return Normal(loc=params["mean"], scale=params["sigma"])


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
        self.A_mean = nn.Parameter(0.001*torch.randn(n))
        self.B_mean = nn.Parameter(0.001*torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))

        self.A_logvar = nn.Parameter(0.001*torch.randn(n))
        self.B_logvar = nn.Parameter(0.001*torch.randn(lb))
        self.bias_logvar = nn.Parameter(torch.zeros(1))

        self.feat_mlp_sigma = nn.Sequential(
            nn.Linear(self.n * self.lb, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.feat_mlp_mu = nn.Sequential(
            nn.Linear(self.n * self.lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.ar_prior_type = "lognormal"
        self.feat_prior_type = "lognormal"
        self.ar_param_key = "mean_logvol"
        self.feat_param_key = "mean_logvol"

    # def bilinear_scalar(self, A, B, X, bias):
    #     tmp = X * A.view(1, self.n, 1)
    #     tmp = tmp.sum(dim=1)
    #     tmp = tmp * B.view(1, self.lb)
    #     return tmp.sum(dim=1) / math.sqrt(self.lb) + bias

    def compute_ar_params(self, past_returns):
        mean_logvol = self.ar_mean_lin(
            past_returns).squeeze(-1).clamp(-10.0, 10.0)  # clamp
        logvar = self.ar_logvar_lin(
            past_returns).squeeze(-1).clamp(-20, 20)     # clamp

        sigma = torch.exp(0.5 * logvar).clamp(min=1e-6,
                                              max=1e3)                     # stricter clamp

        return {
            "mean_logvol": mean_logvol,
            "sigma": sigma
        }

    def compute_feat_params(self, features):
        mean_logvol = self.bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).clamp(-5.0, 5.0)

        logvar = self.bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar)
        # Keeps sigma in reasonable range
        logvar = torch.clamp(logvar, min=-20, max=20)
        sigma = torch.exp(0.5 * logvar)

        return {
            "mean_logvol": mean_logvol,
            "sigma": sigma
        }

    def compute_feat_params_dyn(self, features):
        B = features.shape[0]
        x = features.view(B, -1)
        mu = self.feat_mlp_mu(x).squeeze(-1)
        raw = self.feat_mlp_sigma(x).squeeze(-1)
        sigma = F.softplus(raw) + 1e-6
        return {"mean_logvol": mu, "sigma": sigma}

    def component_mean(self, params: dict) -> torch.Tensor:
        # For lognormal with mu=params["mean_logvol"], sigma => E[X] = exp(mu + 0.5*sigma^2)
        mu = params["mean_logvol"]
        s = params["sigma"]
        return torch.exp(mu + 0.5*(s**2))

    def get_distribution(self, params: dict, component: str = "ar"):
        return LogNormal(loc=params["mean_logvol"], scale=params["sigma"])


class TM_IG_Predictor_param(AbstractMixturePredictor):
    """
    Temporal mixture model with inverse Gaussian components.

    AR component: vol ~ IG(mu_ar, lambda_ar)
    Feature component: vol ~ IG(mu_feat, lambda_feat)

    Ensures:
    - mu > 0 via softplus
    - lambda > 0 via softplus
    """

    def __init__(self, ar_order: int, n: int, lb: int):
        super().__init__(ar_order, n, lb)

        # AR: linear mapping for mu
        self.ar_mean_lin = nn.Linear(ar_order, 1)
        self.raw_lambda_ar = nn.Parameter(
            torch.tensor(0.0))  # softplus(0) ≈ 0.69

        # Feature: bilinear for mu
        self.A_mean = nn.Parameter(torch.randn(n) * 0.01)
        self.B_mean = nn.Parameter(torch.randn(lb) * 0.01)
        self.bias_mean = nn.Parameter(torch.zeros(1))
        self.raw_lambda_feat = nn.Parameter(torch.tensor(0.0))
        self.ar_prior_type = "invgauss"
        self.feat_prior_type = "invgauss"
        self.ar_param_key = "mean"
        self.feat_param_key = "mean"

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        mean = F.softplus(self.ar_mean_lin(
            past_volatility).squeeze(-1)) + 1e-6  # [B]
        lam = F.softplus(self.raw_lambda_ar) + 1e-6
        lam = lam.expand_as(mu)  # [B]
        return {"mean": mean, "lam": lam}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        mean = self.bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)
        mean = F.softplus(mu) + 1e-6  # [B]
        lam = F.softplus(self.raw_lambda_feat) + 1e-6
        lam = lam.expand_as(mu)  # [B]
        return {"mean": mean, "lam": lam}

    def component_mean(self, params: dict) -> torch.Tensor:
        # For inverse Gaussian, E[X] = mu.
        return params["mean"]

    def get_distribution(self, params: dict, component: str = "ar"):
        return InverseGaussian(loc=params["mean"], concentration=params["lam"])


class TM_IG_Predictor(AbstractMixturePredictor):
    """
    Temporal mixture model with inverse-Gaussian components.

    AR component   : v ~ IG(mu_ar,  lambda_ar)
    Feature component: v ~ IG(mu_feat, lambda_feat)

    * mu > 0      enforced via softplus
    * lambda > 0  estimated dynamically (heteroscedastic)
    """

    def __init__(self, ar_order: int, n: int, lb: int, eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # ---------- AR side ----------
        self.ar_mean_lin = nn.Linear(ar_order, 1)        # mu_ar
        self.ar_lambda_lin = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

        # ---------- Feature side ----------
        # mu_feat via bilinear regression
        self.A_mean = nn.Parameter(0.01 * torch.randn(n))
        self.B_mean = nn.Parameter(0.01 * torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))

        # lambda_feat via shallow MLP over flattened features
        self.feat_lambda_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.ar_prior_type = "invgauss"
        self.feat_prior_type = "invgauss"
        self.ar_param_key = "mean"
        self.feat_param_key = "mean"

    # ---- AR component parameters ----

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        """
        Args:
            past_volatility: shape [B, ar_order]
        Returns:
            {'mu': [B], 'lam': [B]}
        """
        mu = F.softplus(self.ar_mean_lin(
            past_volatility)).squeeze(-1)
        lam = F.softplus(self.ar_lambda_lin(
            past_volatility).squeeze(-1)) + self.eps
        return {"mean": mu, "lam": lam}

    # ---- Feature component parameters ----
    def compute_feat_params(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: shape [B, n, lb]
        Returns:
            {'mu': [B], 'lam': [B]}
        """
        # mu_feat (bilinear)
        mu = F.softplus(self.bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean))

        # lambda_feat (MLP)
        x_flat = features.view(features.size(0), -1)          # [B, n*lb]
        lam = F.softplus(self.feat_lambda_net(x_flat).squeeze(-1)) + self.eps
        return {"mean": mu, "lam": lam}

    # ---- Mean of each component (needed for mixture mean / MSE) ----
    def component_mean(self, params: dict) -> torch.Tensor:
        # For inverse Gaussian: E[X] = mu
        return params["mean"]

    def get_distribution(self, params: dict, component: str = "ar"):
        return InverseGaussian(loc=params["mean"], concentration=params["lam"])


class TM_W_Predictor(AbstractMixturePredictor):
    """
    Temporal mixture model with Weibull components using separate networks
    for shape (k) and scale (lam) estimation.
    """

    def __init__(self, ar_order: int, n: int, lb: int, eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # AR networks: separate MLPs for k_ar and lam_ar
        self.ar_k_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ar_lam_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Feature networks: separate MLPs for k_feat and lam_feat
        self.feat_k_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.feat_lam_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # set prior types for KL-divergence
        self.ar_prior_type = "weibull"
        self.feat_prior_type = "weibull"
        self.ar_param_key = "lam"
        self.feat_param_key = "lam"

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        # raw outputs for shape and scale
        raw_k = self.ar_k_net(past_volatility).clamp(-10.0, 10.0)
        raw_lam = self.ar_lam_net(past_volatility).clamp(-10.0, 10.0)

        # softplus ensures positivity, then clamp to reasonable range
        k = F.softplus(raw_k).clamp(min=self.eps, max=10.0).squeeze(-1)
        lam = F.softplus(raw_lam).clamp(min=self.eps, max=10.0).squeeze(-1)

        return {"k": k, "lam": lam}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        # flatten features
        batch_size = features.size(0)
        x_flat = features.view(batch_size, -1)

        # raw outputs for shape and scale
        raw_k = self.feat_k_net(x_flat).clamp(-10.0, 10.0)
        raw_lam = self.feat_lam_net(x_flat).clamp(-10.0, 10.0)

        # softplus ensures positivity, then clamp to reasonable range
        k = F.softplus(raw_k).clamp(min=self.eps, max=10.0).squeeze(-1)
        lam = F.softplus(raw_lam).clamp(min=self.eps, max=10.0).squeeze(-1)

        return {"k": k, "lam": lam}

    def component_mean(self, params: dict) -> torch.Tensor:
        # E[v] = lam * Gamma(1 + 1/k)
        k = params["k"]
        lam = params["lam"]
        inv_k = (1.0 / k).clamp(max=10.0)
        gamma_val = torch.exp(torch.special.gammaln(1.0 + inv_k))
        return lam * gamma_val

    def get_distribution(self, params: dict, component: str = "ar"):
        # return a Weibull distribution with separate parameters
        return Weibull(scale=params["lam"], concentration=params["k"])


class TM_HN_W_Predictor(AbstractMixturePredictor):
    """
    Hinge-Normal + Weibull mixture with separate networks
    for AR mean, AR sigma, Weibull k, and Weibull lambda.
    """

    def __init__(self, ar_order: int, n: int, lb: int, eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # AR mean network

        self.ar_mean_net = nn.Linear(ar_order, 1)

        # AR sigma network
        self.ar_sigma_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Feature k network for Weibull
        self.feat_k_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        # Feature lambda network for Weibull
        self.feat_lam_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # tell the base class which prior to use for each mixture side
        self.ar_prior_type = "normal"
        self.feat_prior_type = "weibull"
        # which key holds the scale parameter for KL
        self.ar_param_key = "sigma"
        self.feat_param_key = "lam"

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        # clamp raw to avoid extremes
        raw_mean = self.ar_mean_net(past_volatility).clamp(-10, 10)
        raw_sigma = self.ar_sigma_net(past_volatility).clamp(-10, 10)

        mean = raw_mean .squeeze(-1).clamp(-10, 10)
        sigma = F.softplus(raw_sigma).clamp(min=self.eps, max=10).squeeze(-1)

        return {"mean": mean, "sigma": sigma}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        x_flat = features.view(features.size(0), -1)
        raw_k = self.feat_k_net(x_flat).clamp(-10, 10)
        raw_lam = self.feat_lam_net(x_flat).clamp(-10, 10)

        k = F.softplus(raw_k).clamp(min=self.eps, max=10).squeeze(-1)
        lam = F.softplus(raw_lam).clamp(min=self.eps, max=10).squeeze(-1)

        return {"k": k, "lam": lam}

    def component_mean(self, params: dict) -> torch.Tensor:
        # if this is the AR side, params has 'mean' & 'sigma'
        if "sigma" in params:
            return params["mean"]
        # otherwise it's the Weibull side
        k = params["k"]
        lam = params["lam"]
        inv_k = (1.0 / k).clamp(max=10)
        gamma_val = torch.exp(torch.special.gammaln(1.0 + inv_k))
        return lam * gamma_val

    def get_distribution(self, params: dict, component: str = "ar"):
        if component == "ar":
            return Normal(loc=params["mean"],
                          scale=params["sigma"].clamp(min=self.eps))
        else:
            return Weibull(scale=params["lam"].clamp(min=self.eps),
                           concentration=params["k"].clamp(min=self.eps))


class TM_HN_IG_Predictor(AbstractMixturePredictor):
    """
    Hinge-Normal + Inverse Gaussian mixture with separate networks
    for AR mean, AR sigma, IG mean, and IG concentration (lambda).
    """

    def __init__(self, ar_order: int, n: int, lb: int, eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # AR mean network
        self.ar_mean_net = nn.Linear(ar_order, 1)
        # AR log-variance network
        self.ar_sigma_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Feature mean network for IG
        self.feat_mean_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        # Feature concentration network for IG
        self.feat_lambda_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        # set prior types for KL-divergence
        self.ar_prior_type = "normal"
        self.feat_prior_type = "invgauss"
        self.ar_param_key = "mean"
        self.feat_param_key = "mean"

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        # clamp raw to avoid extremes
        raw_mean = self.ar_mean_net(past_volatility).clamp(-10, 10)
        raw_sigma = self.ar_sigma_net(past_volatility).clamp(-10, 10)

        mean = F.softplus(raw_mean).squeeze(-1).clamp(-10, 10)
        sigma = F.softplus(raw_sigma).clamp(min=self.eps, max=10).squeeze(-1)

        return {"mean": mean, "sigma": sigma}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        # flatten features
        x_flat = features.view(features.size(0), -1)

        # clamp raw outputs
        raw_mean = self.feat_mean_net(x_flat).clamp(-10.0, 10.0)
        raw_lambda = self.feat_lambda_net(x_flat).clamp(-10.0, 10.0)

        ig_mean = F.softplus(raw_mean).clamp(min=self.eps, max=1e3).squeeze(-1)
        lam = F.softplus(raw_lambda).clamp(min=self.eps, max=1e3).squeeze(-1)

        return {"mean": ig_mean, "lambda": lam}

    def component_mean(self, params: dict) -> torch.Tensor:
        # both Normal and IG component mean = mean
        return params["mean"]

    def get_distribution(self, params: dict, component: str = "ar"):
        if component == "ar":
            return Normal(loc=params["mean"],
                          scale=params["sigma"].clamp(min=self.eps))
        else:
            return InverseGaussian(loc=params["mean"],
                                   concentration=params["lambda"].clamp(min=self.eps))
