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
            prior_loc = torch.distributions.Normal(loc=roll_mean,
                                                   scale=torch.ones_like(roll_mean))
            prior_scale = torch.distributions.LogNormal(-1.0, 0.5)
            return torch.distributions.LogNormal(loc=prior_loc.loc,
                                                 scale=prior_scale.sample())

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
        self.ar_logvar_lin = nn.Linear(ar_order, 1)   # log(σ²)

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

    def compute_feat_params_bi(self, features):
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            return tmp2.sum(dim=1) + bias

        mean_logvol = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).clamp(-5.0, 5.0)

        logvar = bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar)
        # Keeps sigma in reasonable range
        logvar = torch.clamp(logvar, min=-20, max=20)
        sigma = torch.exp(0.5 * logvar)

        return {
            "mean_logvol": mean_logvol,
            "sigma": sigma
        }

    def compute_feat_params(self, features):
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
        self.ar_lambda_lin = nn.Linear(
            ar_order, 1)        # lambda_ar (dynamic)

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
            past_volatility).squeeze(-1)) + self.eps
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
        mu = self.bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)
        mu = F.softplus(mu) + self.eps

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
    Temporal mixture model with Weibull components.

    AR component: v ~ Weibull(k_ar, lam_ar)
    Feature component: v ~ Weibull(k_feat, lam_feat)

    For the AR component, parameters are produced via a small MLP over past_volatility.
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
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # outputs raw_k and raw_lam
        )
        self.ar_prior_type = "weibull"
        self.feat_prior_type = "weibull"
        self.ar_param_key = "lam"
        self.feat_param_key = "lam"
        # Gating: we already have self.ar_gate_lin and bilinear gate from the base class

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        raw = self.ar_net(past_volatility)  # shape [B, 2]
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

    def get_distribution(self, params: dict, component: str = "ar"):
        return Weibull(scale=params["lam"], concentration=params["k"])


class TM_W_Predictor_dynamic(AbstractMixturePredictor):
    """
    Temporal Mixture – Weibull components (ulepszona wersja).

    * AR-component:  v ~ Weibull(k_ar,  lambda_ar)
    * Feat-component: v ~ Weibull(k_feat, lambda_feat)

    Ulepszenia:
    1. Dynamiczne skalowanie parametrów feature-komponentu
    2. Stabilniejsze obliczanie średniej: clamp, lgamma, unikanie overflow
    3. Kompatybilność z GenericMixtureNLL: zwracanie torch.distributions.Weibull
    """

    def __init__(
        self,
        ar_order: int,
        n: int,
        lb: int,
        eps: float = 1e-6,
        k_scale: float = 0.5,
        invk_max: float = 15.0,
    ):
        super().__init__(ar_order, n, lb)
        self.eps = eps
        self.k_scale = k_scale
        self.invk_max = invk_max

        # ------- AR-side MLP -------
        self.ar_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 2),          # raw_k, raw_lambda
        )

        # ------- Feature-side MLP -------
        self.feat_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 2),          # raw_k, raw_lambda
        )
        self.ar_prior_type = "normal"
        self.feat_prior_type = "weibull"
    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _positive(t: torch.Tensor, eps: float) -> torch.Tensor:
        return F.softplus(t) + eps

    # ----- parametrization -------------------------------------------------
    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        raw_k, raw_lam = self.ar_net(past_volatility).split(1, dim=-1)
        k = self._positive(raw_k, self.eps).squeeze(-1)
        lam = self._positive(raw_lam, self.eps).squeeze(-1)
        return {"k": k, "lam": lam}

    def compute_feat_params(self, features_3d: torch.Tensor) -> dict:
        x_flat = features_3d.view(features_3d.size(0), -1)
        raw_k, raw_lam = self.feat_net(x_flat).split(1, dim=-1)
        k = self._positive(raw_k, self.eps).squeeze(-1)
        lam = self._positive(raw_lam, self.eps).squeeze(-1)
        return {"k": k, "lam": lam}

    # ----- moment ----------------------------------------------------------
    def component_mean(self, params: dict) -> torch.Tensor:
        k = params["k"].clamp(min=self.eps)
        lam = params["lam"]
        inv_k = (1.0 / k).clamp(max=self.invk_max)
        gamma_val = torch.exp(torch.lgamma(1.0 + inv_k))
        return lam * gamma_val

    # ----- główny forward --------------------------------------------------
    def forward(
        self,
        past_volatility: torch.Tensor,      # [B, ar_order]
        features_flat: torch.Tensor,        # [B, n*lb]
        roll_mean
    ) -> dict:
        B = features_flat.size(0)
        features_3d = features_flat.view(B, self.n, self.lb)

        # (1) Component parameters
        ar_params = self.compute_ar_params(past_volatility)
        feat_params = self.compute_feat_params(features_3d)

        # (2) Gates
        gate = self.compute_gating_weights(
            past_volatility, features_3d)  # [B,2]

        # (3) Dynamic scaling of feature component
        scale = 1.0 + self.k_scale * gate[:, 1]           # [B]
        feat_params["k"] = feat_params["k"] * scale
        feat_params["lam"] = feat_params["lam"] * scale

        # (4) Means
        ar_mean = self.component_mean(ar_params)
        feat_mean = self.component_mean(feat_params)
        mix_mean = gate[:, 0] * ar_mean + gate[:, 1] * feat_mean

        # (5) Distributions for NLL
        ar_dist = self.get_distribution(ar_params, component="ar")
        feat_dist = self.get_distribution(feat_params, component="feat")

        return {
            "ar_params": ar_params,
            "feat_params": feat_params,
            "gate_weights": gate,          # [B,2]
            "ar_mean": ar_mean,            # [B]
            "feat_mean": feat_mean,        # [B]
            "mixture_mean": mix_mean,      # [B]
            "dists": [ar_dist, feat_dist]  # for GenericMixtureNLL
        }

    # ----- interfejs do GenericMixtureNLL ----------------------------------
    def get_distribution(self, params: dict, component: str = "ar"):
        return Weibull(scale=params["lam"], concentration=params["k"])


class TM_HN_W_Predictor(AbstractMixturePredictor):
    """
    Temporal Mixture Predictor:
      • AR component – Normal(mu, sigma^2) + hinge penalty (mu >= delta)
      • OB component – Weibull(shape, scale)

    Returns:
        {
            'ar_params':   {'mean': mu,      'sigma': sigma},
            'feat_params': {'shape': k,  'scale': lam},
            'gate_weights': [B,2],
            'ar_mean':      mu,
            'feat_mean':    lam * Gamma(1+1/shape),
            'mixture_mean': ...
        }
    """

    def __init__(self, ar_order: int, n: int, lb: int,
                 eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # AR (Normal)
        self.ar_mean_lin = nn.Linear(ar_order, 1)
        self.ar_logvar_lin = nn.Linear(ar_order, 1)

        # OB features (Weibull)
        self.feat_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # raw_shape, raw_scale
        )
        self.ar_prior_type = "normal"
        self.feat_prior_type = "weibull"
        self.ar_param_key = "mean"
        self.feat_param_key = "lam"

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        mu = self.ar_mean_lin(past_volatility).squeeze(-1)
        log_sigma2 = self.ar_logvar_lin(
            past_volatility).squeeze(-1).clamp(-10, 10)
        sigma = torch.exp(0.5 * log_sigma2).clamp(min=1e-4, max=1e2)
        return {"mean": mu, "sigma": sigma}

    def compute_feat_params(self, features: torch.Tensor) -> dict:
        B = features.size(0)
        x = features.view(B, -1)
        raw_shape, raw_scale = self.feat_net(x).split(1, dim=-1)
        shape = F.softplus(raw_shape).squeeze(-1) + self.eps
        scale = F.softplus(raw_scale).squeeze(-1) + self.eps
        return {"k": shape, "lam": scale}

    def component_mean(self, params: dict) -> torch.Tensor:
        if "mean" in params:
            return params["mean"]
        shape, scale = params["k"], params["lam"]
        inv_shape = (1.0 / shape).clamp(max=10.0)
        gamma_val = torch.exp(torch.lgamma(1.0 + inv_shape))
        return scale * gamma_val

    def get_distribution(self, params: dict, component: str = "ar"):
        if component == "ar":
            # Normal component on the AR side
            return Normal(loc=params["mean"], scale=params["sigma"])
        else:                                   # "feat"
            # Weibull component on the order-book side
            return Weibull(scale=params["lam"], concentration=params["k"])


class TM_HN_IG_Predictor(AbstractMixturePredictor):
    """
    Temporal Mixture Predictor:
      • AR component  – Normal(mu, sigma^2)   +  hinge penalty (mu ≥ delta)
      • OB component  – Inverse Gaussian(mean, lambda_)

    Zwraca:
        {
            'ar_params':   {'mean': mu,      'sigma': sigma},
            'feat_params': {'mean': ig_mean, 'lambda': lambda_},
            'gate_weights': [B, 2],
            'ar_mean':        mu,
            'feat_mean':      ig_mean,                # IG mean = mu
            'mixture_mean':   ...
        }
    """

    def __init__(self, ar_order: int, n: int, lb: int,
                 eps: float = 1e-6):
        super().__init__(ar_order, n, lb)
        self.eps = eps

        # ---------- AR (Normal) ----------
        self.ar_mean_lin = nn.Linear(ar_order, 1)
        self.ar_logvar_lin = nn.Linear(ar_order, 1)

        # ---------- OB features (Inverse Gaussian) ----------
        self.feat_net = nn.Sequential(
            nn.Linear(n * lb, 100),
            nn.ReLU(),
            nn.Linear(100, 2)                 # raw_mean, raw_lambda
        )
        self.ar_prior_type = "normal"
        self.feat_prior_type = "invgauss"
        self.ar_param_key = "mean"
        self.feat_param_key = "mean"
    # ----- AR parameters -----

    def compute_ar_params(self, past_volatility: torch.Tensor) -> dict:
        mu = self.ar_mean_lin(past_volatility).squeeze(-1)
        log_sigma2 = self.ar_logvar_lin(
            past_volatility).squeeze(-1).clamp(-10, 10)
        sigma = torch.exp(0.5 * log_sigma2).clamp(min=1e-4, max=1e2)
        return {"mean": mu, "sigma": sigma}

    # ----- OB parameters -----
    def compute_feat_params(self, features: torch.Tensor) -> dict:
        B = features.size(0)
        x = features.view(B, -1)
        raw_mean, raw_lambda = self.feat_net(x).split(1, dim=-1)
        ig_mean = F.softplus(raw_mean).squeeze(-1) + self.eps   # > 0
        lambda_ = F.softplus(raw_lambda).squeeze(-1) + self.eps   # > 0
        return {"mean": ig_mean, "lambda": lambda_}

    # ----- komponentowe średnie -----
    def component_mean(self, params: dict) -> torch.Tensor:
        if "sigma" in params:          # Normal
            return params["mean"]
        # Inverse Gaussian – średnia = mean
        return params["mean"]

    def get_distribution(self, params: dict, component: str = "ar"):
        if component == "ar":
            # Normal component on the AR side
            return Normal(loc=params["mean"], scale=params["sigma"])
        else:                                   # "feat"
            # Inverse-Gaussian component on the order-book side
            return InverseGaussian(loc=params["mean"], concentration=params["lambda"])
