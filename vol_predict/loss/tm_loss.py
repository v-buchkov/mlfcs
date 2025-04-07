from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from vol_predict.loss.abstract_custom_loss import AbstractCustomLoss

class MixtureNormalNLL(AbstractCustomLoss):
    def __init__(self, eps: float = 1e-12,l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

    def forward(
        self, 
        true_returns: torch.Tensor,          
        pred_vol: dict, 
        model: nn.Module                        
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
        gate = pred_vol["gate_weights"]       # [B,2]
        ar = pred_vol["ar_params"]            # dict
        feat = pred_vol["feat_params"]        # dict
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
        l2_penalty = self.compute_l2(model)
        return -log_mix_pdf.mean() + l2_penalty

class HingeNormalMixtureNLL(MixtureNormalNLL):
    def __init__(self, penalty_coef: float = 1.0, delta: float = 0.0, eps: float = 1e-12,l2_coef: float = 0.0):
        super().__init__(eps=eps,l2_coef=l2_coef)
        self.penalty_coef = penalty_coef
        self.delta = delta

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
    def __init__(self, eps: float = 1e-12,l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

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
        gate = pred_vol["gate_weights"]       # shape [B,2]
        ar_params = pred_vol["ar_params"]
        feat_params = pred_vol["feat_params"]

        x = true_returns.clamp(min=self.eps)  # ensure x > 0
        logx = torch.log(x + self.eps)

        mu_ar = ar_params["mean_logvol"]
        sigma_ar = ar_params["sigma"] + self.eps
        mu_feat = feat_params["mean_logvol"]
        sigma_feat = feat_params["sigma"] + self.eps

        # LogNormal log-pdf: log p(x) = -log x - log(sigma*sqrt(2π)) - 0.5*((log x - mu)/sigma)^2
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

        # Combine with gate weights via log-sum-exp
        log_gate0 = torch.log(gate[:, 0] + self.eps)
        log_gate1 = torch.log(gate[:, 1] + self.eps)
        comp0 = log_gate0 + logpdf_ar
        comp1 = log_gate1 + logpdf_feat

        log_mix = torch.logsumexp(torch.stack([comp0, comp1], dim=1), dim=1)
        nll = -log_mix.mean()
        l2_penalty = self.compute_l2(model)
        return nll + l2_penalty


class MixtureInverseGaussianNLL(AbstractCustomLoss):
    def __init__(self, eps: float = 1e-12,l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

    def forward(
        self, true_returns: torch.Tensor, pred_vol: dict, model: nn.Module
    ) -> torch.Tensor:
        """
        Assumes pred_vol contains:
            - "gate_weights": [B, 2]
            - "ar_params": dict with keys "mu" and "lam"
            - "feat_params": dict with keys "mu" and "lam"
        """
        gate = pred_vol["gate_weights"]
        ar_params = pred_vol["ar_params"]
        feat_params = pred_vol["feat_params"]

        x = true_returns.clamp(min=self.eps)
        # For Inverse Gaussian, the pdf is:
        # log p(x|mu, lam) = 0.5*log(lam/(2π*x^3)) - (lam*(x-mu)^2) / (2*mu^2*x)
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
            - (lam_feat * (x - mu_feat)**2) / (2.0 * (mu_feat**2) * x + self.eps)
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
    def __init__(self, eps: float = 1e-12,l2_coef: float = 0.0):
        super().__init__(l2_coef=l2_coef)
        self.eps = eps

    def forward(
        self, true_returns: torch.Tensor, pred_vol: dict, model: nn.Module
    ) -> torch.Tensor:
        """
        Assumes pred_vol contains:
            - "gate_weights": [B, 2]
            - "ar_params": dict with keys "k" and "lam"
            - "feat_params": dict with keys "k" and "lam"
        """
        gate = pred_vol["gate_weights"]
        ar_params = pred_vol["ar_params"]
        feat_params = pred_vol["feat_params"]

        x = true_returns.clamp(min=self.eps)

        # Weibull log-pdf:
        # log f(x|k, lam) = log(k) - log(lam) + (k-1)*log(x/lam) - (x/lam)^k
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
        return nll
