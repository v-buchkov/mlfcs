from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss

# Currently we are using mostly the same loss for all mixture models.
# This is a generic loss that can be used for any mixture of distributions.

class GenericMixtureNLL(AbstractCustomLoss):
    """
    Universal negative log-likelihood for a mixture of K distributions.
    Requires the predictor to return:
        - "gate_weights"  – tensor [B,K] with component weights
        - "dists"         – list of length K of torch.distributions.* objects
    Optionally, you can include a CRPS penalty (crps_weight > 0) and L2 (l2_coef > 0).
    """

    def __init__(self, crps_weight: float = 0.0, kl_weight: float = 0, eps: float = 1e-12, l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps
        self.crps_weight = crps_weight
        self._model = None  # backup storage of the model
        self.kl_weight = kl_weight
    # allows injecting a model from the training loop

    def set_model(self, model: nn.Module) -> None:
        self._model = model

    def forward(self,
                true_y: torch.Tensor,
                pred_vol: dict,
                model: nn.Module = None) -> torch.Tensor:

        if model is None:
            model = self._model

        gate = pred_vol["gate_weights"]      # [B,K]
        dists = pred_vol["dists"]            # list length K
        priors = pred_vol.get("prior_dists", None)

        # log g_k(x) + log p_k(x)
        log_comp = torch.stack(
            [torch.log(gate[:, k] + self.eps) + dists[k].log_prob(true_y)
             for k in range(len(dists))],
            dim=1)  # [B,K]

        nll = -torch.logsumexp(log_comp, dim=1).mean()
        # ------------ KL ---------------
        kl_term = 0.0
        if priors is not None and self.kl_weight > 0:
            kl_each = [torch.distributions.kl.kl_divergence(dists[k], priors[k]).mean()
                       for k in range(len(dists))]
            kl_term = sum(kl_each) / len(kl_each)

        # ─── Continuous Ranked Probability Score ───
        if self.crps_weight > 0.0:
            S = 100  # Fewer samples for efficiency
            with torch.no_grad():
                # Sample from the predictive distribution for each batch element
                samples = torch.stack(
                    [d.rsample((S,)) for d in dists],
                    dim=2)  # [S, B, K]

                # Compute mixture samples using the gating (softmax) weights
                mix_samples = (samples * gate.unsqueeze(0)
                               ).sum(dim=2)  # [S, B]

                # Term 1: E|X - y|
                term1 = torch.abs(mix_samples - true_y).mean(dim=0)  # [B]

                # Term 2: 0.5 * E|X - X'| using the sorting trick
                sorted_samples, _ = torch.sort(mix_samples, dim=0)  # [S, B]
                idx = torch.arange(
                    1, S + 1, device=mix_samples.device).float().unsqueeze(1)  # [S, 1]
                weight = 2 * (idx - 1) - (S - 1)  # [S, 1]
                term2 = (sorted_samples * weight).sum(dim=0) / \
                    (S * (S - 1))  # [B]

                crps = (term1 - term2).mean()  # scalar

            nll = nll + self.crps_weight * crps

        return nll + self.compute_l2(model) + self.kl_weight * kl_term

# Losses below are a relict of the old implementation.

class MixtureNormalNLL(AbstractCustomLoss):
    def __init__(self, eps: float = 1e-12, l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps
        self._model = None

    def set_model(self, model: nn.Module) -> None:
        self._model = model

    def forward(
        self,
        true_returns: torch.Tensor,
        pred_vol: dict,
        model: nn.Module = None
    ) -> torch.Tensor:
        """
        Args:
            true_returns: true volatility values (target), shape [B]
            pred_vol: dictionary with keys:
                - "gate_weights": [B, 2]
                - "ar_params": dict with "mean", "sigma"
                - "feat_params": dict with "mean", "sigma"
        Returns:
            scalar loss (NLL)
        """
        if model is None:
            model = self._model

        gate = pred_vol["gate_weights"] 
        ar = pred_vol["ar_params"]   
        feat = pred_vol["feat_params"]      
        x = true_returns.clamp(min=self.eps)  # avoid log(0)

        mu_ar = ar["mean"]
        sigma_ar = ar["sigma"] + self.eps
        mu_feat = feat["mean"]
        sigma_feat = feat["sigma"] + self.eps

        # logpdf for Normal distribution
        logpdf_ar = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma_ar)
            - 0.5 * ((x - mu_ar) / sigma_ar) ** 2
        )
        logpdf_feat = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma_feat)
            - 0.5 * ((x - mu_feat) / sigma_feat) ** 2
        )

        # Mixture log likelihood: log(g1 * p1 + g2 * p2)
        log_mix_pdf = torch.logsumexp(
            torch.stack([
                torch.log(gate[:, 0] + self.eps) + logpdf_ar,
                torch.log(gate[:, 1] + self.eps) + logpdf_feat
            ], dim=1),
            dim=1
        )
        l2_penalty = self.compute_l2(model) if model is not None else 0.0
        return -log_mix_pdf.mean() + l2_penalty

class HingeNormalMixtureNLL(MixtureNormalNLL):
    def __init__(self, penalty_coef: float = 1.0, delta: float = 0.0, eps: float = 1e-12, l2_coef: float = 0.0):
        super().__init__(eps=eps, l2_coef=l2_coef)
        self.penalty_coef = penalty_coef
        self.delta = delta

    def set_model(self, model: nn.Module) -> None:
        self._model = model

    def forward(
        self, true_returns: torch.Tensor, pred_vol: dict, model: nn.Module
    ) -> torch.Tensor:
        base_loss = super().forward(true_returns, pred_vol)

        # Penalty for negative mean predictions
        ar_mean = pred_vol["ar_mean"]
        feat_mean = pred_vol["feat_mean"]

        penalty = (
            torch.relu(self.delta - ar_mean) +
            torch.relu(self.delta - feat_mean)
        ).mean()

        return base_loss + self.penalty_coef * penalty


class MixtureLogNormalNLL(AbstractCustomLoss):
    def __init__(self, eps: float = 1e-12, l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

    def set_model(self, model: nn.Module) -> None:
        self._model = model

    def forward(
        self, true_returns: torch.Tensor, pred_vol: dict, model: nn.Module
    ) -> torch.Tensor:
        """
        Assumes pred_vol contains:
            - "gate_weights": [B, 2]
            - "ar_params": dict with keys "mean_logvol" and "sigma"
            - "feat_params": dict with keys "mean_logvol" and "sigma"
        true_returns should be positive (volatility).
        """
        if model is None:
            model = self._model
        gate = pred_vol["gate_weights"]  # shape [B,2]
        ar_params = pred_vol["ar_params"]
        feat_params = pred_vol["feat_params"]

        x = true_returns.clamp(min=self.eps)  # ensure x > 0
        logx = torch.log(x + self.eps)

        mu_ar = ar_params["mean_logvol"]
        sigma_ar = ar_params["sigma"] + self.eps
        mu_feat = feat_params["mean_logvol"]
        sigma_feat = feat_params["sigma"] + self.eps

        # LogNormal log-pdf
        logpdf_ar = (
            - logx
            - torch.log(sigma_ar * math.sqrt(2.0 * math.pi) + self.eps)
            - 0.5 * ((logx - mu_ar) / sigma_ar) ** 2
        )
        logpdf_feat = (
            - logx
            - torch.log(sigma_feat * math.sqrt(2.0 * math.pi) + self.eps)
            - 0.5 * ((logx - mu_feat) / sigma_feat) ** 2
        )

        log_gate0 = torch.log(gate[:, 0] + self.eps)
        log_gate1 = torch.log(gate[:, 1] + self.eps)
        comp0 = log_gate0 + logpdf_ar
        comp1 = log_gate1 + logpdf_feat

        log_mix = torch.logsumexp(torch.stack([comp0, comp1], dim=1), dim=1)
        nll = -log_mix.mean()
        l2_penalty = self.compute_l2(model)
        variance_penalty_weight = 0.01
        variance_penalty = ((sigma_ar**2).mean() +
                            (sigma_feat**2).mean()) * variance_penalty_weight
        return nll + variance_penalty + l2_penalty


class MixtureInverseGaussianNLL(AbstractCustomLoss):
    def __init__(self, eps: float = 1e-12, l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

    def set_model(self, model: nn.Module) -> None:
        self._model = model

    def forward(
        self, true_returns: torch.Tensor, pred_vol: dict, model: nn.Module
    ) -> torch.Tensor:
        """
        Assumes pred_vol contains:
            - "gate_weights": [B, 2]
            - "ar_params": dict with keys "mu" and "lam"
            - "feat_params": dict with keys "mu" and "lam"
        """
        if model is None:
            model = self._model
        gate = pred_vol["gate_weights"]
        ar_params = pred_vol["ar_params"]
        feat_params = pred_vol["feat_params"]

        x = true_returns.clamp(min=self.eps)
        # For Inverse Gaussian, the pdf is:
        # log p(x|mu, lam) = 0.5*log(lam/(2pi*x^3)) - (lam*(x-mu)^2) / (2*mu^2*x)
        mu_ar = ar_params["mu"]
        lam_ar = ar_params["lam"] + self.eps
        mu_feat = feat_params["mu"]
        lam_feat = feat_params["lam"] + self.eps

        logpdf_ar = (
            0.5 * torch.log(lam_ar / (2.0 * math.pi * (x**3) + self.eps))
            - (lam_ar * (x - mu_ar)**2) / (2.0 * (mu_ar**2) * x + self.eps)
        )
        logpdf_feat = (
            0.5 * torch.log(lam_feat / (2.0 * math.pi * (x**3) + self.eps))
            - (lam_feat * (x - mu_feat)**2) /
            (2.0 * (mu_feat**2) * x + self.eps)
        )

        log_gate0 = torch.log(gate[:, 0] + self.eps)
        log_gate1 = torch.log(gate[:, 1] + self.eps)
        comp0 = log_gate0 + logpdf_ar
        comp1 = log_gate1 + logpdf_feat

        log_mix = torch.logsumexp(torch.stack([comp0, comp1], dim=1), dim=1)
        nll = -log_mix.mean()
        l2_penalty = self.compute_l2(model)
        return nll + l2_penalty


class MixtureWeibullNLL(AbstractCustomLoss):
    def __init__(self, eps: float = 1e-12, l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

    def set_model(self, model: nn.Module) -> None:
        self._model = model

    def forward(
        self, true_returns: torch.Tensor, pred_vol: dict, model: nn.Module
    ) -> torch.Tensor:
        """
        Assumes pred_vol contains:
            - "gate_weights": [B, 2]
            - "ar_params": dict with keys "k" and "lam"
            - "feat_params": dict with keys "k" and "lam"
        """
        if model is None:
            model = self._model
        gate = pred_vol["gate_weights"]
        ar_params = pred_vol["ar_params"]
        feat_params = pred_vol["feat_params"]

        x = true_returns.clamp(min=self.eps)

        # Weibull log-pdf
        k_ar = ar_params["k"] + self.eps
        lam_ar = ar_params["lam"] + self.eps
        k_feat = feat_params["k"] + self.eps
        lam_feat = feat_params["lam"] + self.eps

        logpdf_ar = (
            torch.log(k_ar)
            - torch.log(lam_ar)
            + (k_ar - 1.0) * torch.log(x / lam_ar + self.eps)
            - (x / lam_ar).pow(k_ar)
        )
        logpdf_feat = (
            torch.log(k_feat)
            - torch.log(lam_feat)
            + (k_feat - 1.0) * torch.log(x / lam_feat + self.eps)
            - (x / lam_feat).pow(k_feat)
        )

        log_gate0 = torch.log(gate[:, 0] + self.eps)
        log_gate1 = torch.log(gate[:, 1] + self.eps)
        comp0 = log_gate0 + logpdf_ar
        comp1 = log_gate1 + logpdf_feat

        log_mix = torch.logsumexp(torch.stack([comp0, comp1], dim=1), dim=1)
        nll = -log_mix.mean()
        l2_penalty = self.compute_l2(model)
        return nll + l2_penalty


class MixtureHingeNormalWeibullNLL(GenericMixtureNLL):
    """
    Mixture of a Normal (AR) and Weibull (features) with
    - KL-divergence regularisation (handled by GenericMixtureNLL)
    - hinge penalty to keep the component means ≥ delta.
    """

    def __init__(
        self,
        penalty_coef: float = 1.0,
        delta: float = 0.0,
        eps: float = 1e-12,
        crps_weight: float = 0.0,
        kl_weight: float = 1e-3,
        l2_coef: float = 0.0,
    ):
        super().__init__(
            eps=eps, crps_weight=crps_weight, kl_weight=kl_weight, l2_coef=l2_coef
        )
        self.penalty_coef = penalty_coef
        self.delta = delta

    def forward(self, true_returns: torch.Tensor, pred_vol: dict, model=None):
        base = super().forward(true_returns, pred_vol, model)          # NLL + KL + L2 + CRPS
        ar_mean = pred_vol["ar_mean"]
        feat_mean = pred_vol["feat_mean"]
        hinge = (F.relu(self.delta - ar_mean) +
                 F.relu(self.delta - feat_mean)).mean()
        return base + self.penalty_coef * hinge


class MixtureHingeNormalInvGaussianNLL(GenericMixtureNLL):
    """
    Mixture of a Normal (AR) and Inverse-Gaussian (features) with
    - KL-divergence regularisation (handled by GenericMixtureNLL)
    - hinge penalty to keep the component means ≥ delta.
    """

    def __init__(
        self,
        penalty_coef: float = 1.0,
        delta: float = 0.0,
        eps: float = 1e-12,
        crps_weight: float = 0.0,
        kl_weight: float = 1e-3,
        l2_coef: float = 0.0,
    ):
        super().__init__(
            eps=eps, crps_weight=crps_weight, kl_weight=kl_weight, l2_coef=l2_coef
        )
        self.penalty_coef = penalty_coef
        self.delta = delta

    def forward(self, true_returns: torch.Tensor, pred_vol: dict, model=None):
        base = super().forward(true_returns, pred_vol, model)          # NLL + KL + L2 + CRPS
        ar_mean = pred_vol["ar_mean"]
        feat_mean = pred_vol["feat_mean"]
        hinge = (F.relu(self.delta - ar_mean) +
                 F.relu(self.delta - feat_mean)).mean()
        return base + self.penalty_coef * hinge
