from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import math
from abc import ABC


from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractPredictor(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.hidden = None
        self.memory = None

    def forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(past_returns, features, *args, **kwargs)

    def __call__(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.forward(past_returns, features, *args, **kwargs)

    @abstractmethod
    def _forward(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class TemporalMixtureLogNormalPredictorBilinear(AbstractPredictor):
    """
    A temporal mixture model with log-normal components:

      - AR component (depends on past_returns): 
          log(vol) ~ Normal(ar_mean_logvol, ar_sigma^2)
      - Feature component (depends on a bilinear function of features): 
          log(vol) ~ Normal(feat_mean_logvol, feat_sigma^2)

    The gating function is also a mixture of:
      - AR gate logit: linear in past_returns
      - Feature gate logit: bilinear in features

    Then we take a softmax to form gating weights.
    """

    def __init__(
        self,
        ar_order: int,   # number of past lags used for AR
        n: int,          # number of "feature dims" per time slice
        lb: int,         # how many time slices in the look-back window for features
    ):
        """
        Args:
            ar_order: how many past returns/vols are used for AR part
            n:        dimension of each feature slice
            lb:       how many time slices of features we have
        """
        super().__init__()

        self.ar_order = ar_order
        self.n = n
        self.lb = lb

        # ============ AR COMPONENT (MEAN & LOGVAR) ============
        #
        #  ar_mean_lin:  linear => [ar_order] -> [1]
        #  ar_logvar_lin: same, if we want separate variance or log-variance
        #
        self.ar_mean_lin = nn.Linear(ar_order, 1)    # for mu_AR
        self.ar_logvar_lin = nn.Linear(ar_order, 1)  # for logvar_AR

        # ============ FEATURE COMPONENT (BILINEAR) ============
        #
        # We'll do a rank-1 bilinear form for the mean:
        #   feat_mean = A_mean^T * X * B_mean
        #   (where X is shape [n, lb])
        #
        # So we only store (A_mean in R^n) and (B_mean in R^lb).
        # Similarly for logvar:
        #   feat_logvar = A_logvar^T * X * B_logvar
        #
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))  # optional bias

        self.A_logvar = nn.Parameter(torch.randn(n))
        self.B_logvar = nn.Parameter(torch.randn(lb))
        self.bias_logvar = nn.Parameter(torch.zeros(1))

        # ============ GATING FUNCTION ============
        #
        #  We'll produce two logits:
        #   1) AR gate logit => linear in [ar_order]
        #   2) FEAT gate logit => bilinear in X
        #
        # Then do a softmax over them => gating weights in [0,1] that sum to 1
        #
        self.ar_gate_lin = nn.Linear(ar_order, 1)  # AR gating
        # Feature gating (bilinear)
        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

    def _forward(
        self,
        past_returns: torch.Tensor,   # shape [batch_size, ar_order]
        features: torch.Tensor,       # shape [batch_size, n, lb]
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Return the mixture's *mean* forecast E[v_h]. 
        That is  g_AR * E[v_h|AR] + g_FEAT * E[v_h|Feat].
        (where E[v_h] for log-normal is exp(mu + 0.5 sigma^2))

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
        Returns:
            mixture_mean: [batch_size]
        """
        # =============== AR COMPONENT ===============
        # Mean of log(vol):  shape [batch_size, 1]
        ar_mean_logvol = self.ar_mean_lin(past_returns)
        # Log-var:          shape [batch_size, 1]
        ar_logvar = self.ar_logvar_lin(past_returns)
        # => sigma for AR
        # shape [batch_size, 1] (this is just algebraic formula using log properties)
        ar_sigma = torch.exp(0.5 * ar_logvar)

        # =============== FEATURE COMPONENT (BILINEAR) ===============
        # We'll define a small helper that does A^T X B for each batch item:
        def bilinear_scalar(A, B, X, bias):
            """
            A in R^n
            B in R^lb
            X in R^[batch_size, n, lb]
            bias in R^[1]
            Return => shape [batch_size]
            """
            # multiply along feature dimension:
            # scale each "row" by A (for each batch)
            tmp = X * A.view(1, self.n, 1)
            # sum over features => shape [batch_size, lb]
            tmp_sumF = tmp.sum(dim=1)

            # multiply along time dimension by B
            tmp2 = tmp_sumF * B.view(1, self.lb)  # shape [batch_size, lb]
            # here similiarly we sum over the second dimension, that is here lb => shape [batch_size]
            val = tmp2.sum(dim=1)
            return val + bias  # shape [batch_size]

        # (a) mean of log-vol => shape [batch_size]
        feat_mean_logvol = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)
        # expand dims => shape [batch_size, 1]
        feat_mean_logvol = feat_mean_logvol.unsqueeze(
            1)  # for further technical reasons

        # (b) logvar => shape [batch_size]
        # here we get the logvariance of feature based component in the model
        feat_logvar = bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar)
        feat_logvar = feat_logvar.unsqueeze(1)

        # and here we transform the logvariance to variance once again using standard algebraic properties

        feat_sigma = torch.exp(0.5 * feat_logvar)  # shape [batch_size, 1]

        # =============== GATING ===============
        # AR gating logit => shape [batch_size, 1]
        gate_ar_score = self.ar_gate_lin(past_returns)

        # Feature gating logit => shape [batch_size]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate)
        feat_gate_score = feat_gate_score.unsqueeze(
            1)  # => shape [batch_size, 1]

        # stack => shape [batch_size, 2], then softmax
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # => [batch_size, 2]

        # gate_weights[:,0] => AR mixture weight
        # gate_weights[:,1] => FEAT mixture weight

        # =============== COMPUTE MIXTURE'S MEAN ===============
        # Here we are using the assumption of log normality of distribution,
        # mark that its going to be different for other distributions

        # For a log-normal random variable with parameters (mu, sigma),
        # E[v] = exp(mu + 0.5*sigma^2).
        # shape => [batch_size]
        ar_comp_mean = torch.exp(
            ar_mean_logvol.squeeze(-1) + 0.5 * (ar_sigma.squeeze(-1) ** 2))
        feat_comp_mean = torch.exp(
            feat_mean_logvol.squeeze(-1) + 0.5 * (feat_sigma.squeeze(-1) ** 2))

        mixture_mean = gate_weights[:, 0] * \
            ar_comp_mean + gate_weights[:, 1] * feat_comp_mean

        return mixture_mean

    def negative_log_likelihood(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        target_vol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional: mixture log-likelihood for training.
        negative_log_likelihood = - sum(log(gate0 * LNpdf_AR + gate1 * LNpdf_Feat)).
        (LNpdf_*) is the lognormal pdf for each component.

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
            target_vol:   [batch_size]
        Returns:
            nll: scalar
        """
        # The logic up until "Mixture logpdf" is the same as in forward

        # --- AR component params ---
        ar_mean_logvol = self.ar_mean_lin(past_returns)  # [batch_size, 1]
        ar_logvar = self.ar_logvar_lin(past_returns)     # [batch_size, 1]
        ar_sigma = torch.exp(0.5 * ar_logvar)            # [batch_size, 1]

        # --- Feature component params (bilinear) ---
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            val = tmp2.sum(dim=1)
            return val + bias

        feat_mean_logvol = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).unsqueeze(1)
        feat_logvar = bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar).unsqueeze(1)
        feat_sigma = torch.exp(0.5 * feat_logvar)

        # --- Gating function ---
        gate_ar_score = self.ar_gate_lin(past_returns)  # [batch_size, 1]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate).unsqueeze(1)
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # [batch_size, 2]

        # --- Mixture logpdf ---
        # log-normal pdf:
        # log p(x) = - log x - log(sigma sqrt(2pi)) - 0.5 ((ln x - mu)/sigma)^2
        eps = 1e-12
        # we are bounding x from below to deal with indefinitness of log at x=0
        x = torch.clamp(target_vol, min=eps)
        logx = torch.log(x)

        ar_log_pdf = (
            - logx
            - torch.log(ar_sigma.squeeze(-1) * (2.0 * pi)**0.5 + eps)
            - 0.5 * ((logx - ar_mean_logvol.squeeze(-1)) /
                     (ar_sigma.squeeze(-1) + eps))**2
        )
        feat_log_pdf = (
            - logx
            - torch.log(feat_sigma.squeeze(-1) * (2.0 * pi)**0.5 + eps)
            - 0.5 * ((logx - feat_mean_logvol.squeeze(-1)) /
                     (feat_sigma.squeeze(-1) + eps))**2
        )

        comp0 = torch.log(gate_weights[:, 0] + eps) + ar_log_pdf
        comp1 = torch.log(gate_weights[:, 1] + eps) + feat_log_pdf
        # log mixture = logsumexp( comp0, comp1 )
        log_mix_pdf = torch.logsumexp(
            torch.stack([comp0, comp1], dim=1), dim=1)

        nll = - log_mix_pdf.mean()
        return nll


class TemporalMixtureReverseGaussianFixed(AbstractPredictor):
    """
    A temporal mixture model with reverse-gaussian components:

      - AR component (depends on past_returns): 
          vol ~ reverse_gaussian(ar_mean, ar_lambda)
      - Feature component (depends on a bilinear function of features): 
          log(vol) ~ Normal(feat_mean, feat_lambda)

    The gating function is also a mixture of:
      - AR gate logit: linear in past_returns
      - Feature gate logit: bilinear in features

    Then we take a softmax to form gating weights.

    In this class we take lambda parameter to be global (different for feat and ar). 
    It is simpler approach assuming some sort of stationarity / homoskedasticity. It prevents overfitting.
    """

    def __init__(
        self,
        ar_order: int,   # number of past lags used for AR
        n: int,          # number of "feature dims" per time slice
        lb: int,         # how many time slices in the look-back window for features
    ):
        """
        Args:
            ar_order: how many past returns/vols are used for AR part
            n:        dimension of each feature slice
            lb:       how many time slices of features we have
        """
        super().__init__()

        self.ar_order = ar_order
        self.n = n
        self.lb = lb

        # ============ AR COMPONENT (MEAN & global LAMBDA) ============
        #
        #  ar_mean_lin:  linear => [ar_order] -> [1]
        #  ar_logvar_lin: same, if we want separate variance or log-variance
        #
        self.ar_mean_lin = nn.Linear(ar_order, 1)    # for mu_AR
        self.lambda_ar = nn.Parameter(torch.tensor(1.0))

        # ============ FEATURE COMPONENT (BILINEAR) ============
        #
        # We'll do a rank-1 bilinear form for the mean:
        #   feat_mean = A_mean^T * X * B_mean
        #   (where X is shape [n, lb])
        #
        # So we only store (A_mean in R^n) and (B_mean in R^lb).
        # Similarly for logvar:
        #   feat_logvar = A_logvar^T * X * B_logvar
        #
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))  # optional bias

        self.lambda_feat = nn.Parameter(torch.tensor(1.0))

        # ============ GATING FUNCTION ============
        #
        #  We'll produce two logits:
        #   1) AR gate logit => linear in [ar_order]
        #   2) FEAT gate logit => bilinear in X
        #
        # Then do a softmax over them => gating weights in [0,1] that sum to 1
        #
        self.ar_gate_lin = nn.Linear(ar_order, 1)  # AR gating

        # Feature gating (bilinear)
        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

    def _forward(
        self,
        past_returns: torch.Tensor,   # shape [batch_size, ar_order]
        features: torch.Tensor,       # shape [batch_size, n, lb]
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Return the mixture's *mean* forecast E[v_h]. 
        That is  g_AR * E[v_h|AR] + g_FEAT * E[v_h|Feat].
        (where E[v_h] for log-normal is exp(mu + 0.5 sigma^2))

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
        Returns:
            mixture_mean: [batch_size]
        """

        batch_size = past_returns.shape[0]

        # =============== AR COMPONENT ===============
        # Mean of log(vol):  shape [batch_size, 1]
        ar_mean = self.ar_mean_lin(past_returns)

        # =============== FEATURE COMPONENT (BILINEAR) ===============
        # We'll define a small helper that does A^T X B for each batch item:
        def bilinear_scalar(A, B, X, bias):
            """
            A in R^n
            B in R^lb
            X in R^[batch_size, n, lb]
            bias in R^[1]
            Return => shape [batch_size]
            """
            # multiply along feature dimension:
            # scale each "row" by A (for each batch)
            tmp = X * A.view(1, self.n, 1)
            # sum over features => shape [batch_size, lb]
            tmp_sumF = tmp.sum(dim=1)

            # multiply along time dimension by B
            tmp2 = tmp_sumF * B.view(1, self.lb)  # shape [batch_size, lb]
            # here similiarly we sum over the second dimension, that is here lb => shape [batch_size]
            val = tmp2.sum(dim=1)
            return val + bias  # shape [batch_size]

        # (a) mean of log-vol => shape [batch_size]
        feat_mean = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)
        # expand dims => shape [batch_size, 1]
        feat_mean = feat_mean.unsqueeze(
            1)  # for further technical reasons

        # =============== GATING ===============
        # AR gating logit => shape [batch_size, 1]
        gate_ar_score = self.ar_gate_lin(past_returns)

        # Feature gating logit => shape [batch_size]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate)
        feat_gate_score = feat_gate_score.unsqueeze(
            1)  # => shape [batch_size, 1]

        # stack => shape [batch_size, 2], then softmax
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # => [batch_size, 2]

        # gate_weights[:,0] => AR mixture weight
        # gate_weights[:,1] => FEAT mixture weight

        # =============== COMPUTE MIXTURE'S MEAN ===============
        # Here we are using the assumption of reverse gaussian distribution
        # shape => [batch_size]

        # Inverse Gaussian expectation: E[X] = μ

        ar_comp_mean = ar_mean.squeeze(-1)
        feat_comp_mean = feat_mean.squeeze(-1)

        mixture_mean = gate_weights[:, 0] * \
            ar_comp_mean + gate_weights[:, 1] * feat_comp_mean

        return mixture_mean

    def negative_log_likelihood(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        target_vol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional: mixture log-likelihood for training.
        negative_log_likelihood = - sum(log(gate0 * LNpdf_AR + gate1 * LNpdf_Feat)).
        (LNpdf_*) is the lognormal pdf for each component.

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
            target_vol:   [batch_size]
        Returns:
            nll: scalar
        """
        # The logic up until "Mixture logpdf" is the same as in forward

        batch_size = past_returns.shape[0]

        # --- AR component params ---
        ar_mean = self.ar_mean_lin(past_returns)  # [batch_size, 1]

        lambda_ar = self.lambda_ar.expand(batch_size, 1)

        # --- Feature component params (bilinear) ---
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            val = tmp2.sum(dim=1)
            return val + bias

        def inverse_gaussian_logpdf(x, mu, lam, eps=1e-12):
            x = torch.clamp(x, min=eps)
            term1 = 0.5 * torch.log(lam / (2 * torch.pi * x**3))
            term2 = -(lam * (x - mu)**2) / (2 * mu**2 * x)
            return term1 + term2

        feat_mean = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).unsqueeze(1)

        lambda_feat = self.lambda_feat.expand(batch_size, 1)

        # --- Gating function ---
        gate_ar_score = self.ar_gate_lin(past_returns)  # [batch_size, 1]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate).unsqueeze(1)
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # [batch_size, 2]

        # --- Mixture logpdf ---
        # log-inverse_gaussian pdf:
        eps = 1e-12
        # we are bounding x from below to deal with indefinitness of log at x=0
        x = torch.clamp(target_vol, min=eps)
        logx = torch.log(x)

        ar_log_pdf = inverse_gaussian_logpdf(x, ar_mean, lambda_ar)
        feat_log_pdf = inverse_gaussian_logpdf(x, feat_mean, lambda_feat)

        comp0 = torch.log(gate_weights[:, 0] + eps) + ar_log_pdf
        comp1 = torch.log(gate_weights[:, 1] + eps) + feat_log_pdf
        # log mixture = logsumexp( comp0, comp1 )
        log_mix_pdf = torch.logsumexp(
            torch.stack([comp0, comp1], dim=1), dim=1)

        nll = - log_mix_pdf.mean()
        return nll


class TemporalMixtureNormalPredictorBilinear(AbstractPredictor):
    """
    A temporal mixture model with Gaussian (normal) components:

      - AR component (depends on past_returns):
          vol ~ Normal(ar_mean, ar_sigma^2)
      - Feature component (depends on a bilinear function of features):
          vol ~ Normal(feat_mean, feat_sigma^2)

    The gating function is also a mixture of:
      - AR gate logit: linear in past_returns
      - Feature gate logit: bilinear in features

    Then we take a softmax to form gating weights.

    Additionally, we add a penalty for negative mean predictions
    (as the paper suggests for normalizing the model on positive volatility).
    """

    def __init__(
        self,
        ar_order: int,    # number of past lags used for AR
        n: int,           # number of "feature dims" per time slice
        lb: int,          # how many time slices in the look-back window for features
        penalty_coef: float = 1.0,  # coefficient for penalizing negative means
        delta: float = 0.0
    ):
        """
        Args:
            ar_order:   how many past returns/vols are used for AR part
            n:          dimension of each feature slice
            lb:         how many time slices of features we have
            penalty_coef: weight for the hinge penalty on negative means
        """
        super().__init__()

        self.ar_order = ar_order
        self.n = n
        self.lb = lb
        self.penalty_coef = penalty_coef
        self.delta = delta

        # ============ AR COMPONENT (MEAN & LOGVAR) ============
        #
        # We'll have a linear layer for AR mean:  mu_AR
        self.ar_mean_lin = nn.Linear(ar_order, 1)    # for mu_AR

        # We'll have a linear layer for log-variance: log_sigma^2_AR
        self.ar_logvar_lin = nn.Linear(ar_order, 1)  # for logvar_AR

        # ============ FEATURE COMPONENT (BILINEAR) ============
        #
        # We'll do a rank-1 bilinear form for the mean:
        #   feat_mean = A_mean^T * X * B_mean
        #
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))  # optional bias

        # Similarly for the log-variance:
        #   feat_logvar = A_logvar^T * X * B_logvar
        #
        self.A_logvar = nn.Parameter(torch.randn(n))
        self.B_logvar = nn.Parameter(torch.randn(lb))
        self.bias_logvar = nn.Parameter(torch.zeros(1))

        # ============ GATING FUNCTION ============
        #
        #  We'll produce two logits:
        #   1) AR gate logit => linear in [ar_order]
        #   2) FEAT gate logit => bilinear in X
        # Then do a softmax => gating weights in [0,1] that sum to 1
        #
        self.ar_gate_lin = nn.Linear(ar_order, 1)  # AR gating
        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

    def _forward(
        self,
        past_returns: torch.Tensor,   # shape [batch_size, ar_order]
        features: torch.Tensor,       # shape [batch_size, n, lb]
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Return the mixture's *mean* forecast E[v_h].
        That is  g_AR * mu_AR + g_FEAT * mu_FEAT
        (Because for a Normal distribution, E[X] = mu.)

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
        Returns:
            mixture_mean: [batch_size]
        """
        # =============== AR COMPONENT ===============
        # AR mean => shape [batch_size, 1]
        ar_mean = self.ar_mean_lin(past_returns)

        # =============== FEATURE COMPONENT (BILINEAR) ===============
        def bilinear_scalar(A, B, X, bias):
            """
            A in R^n
            B in R^lb
            X in R^[batch_size, n, lb]
            bias in R^[1]
            Return => shape [batch_size]
            """
            tmp = X * A.view(1, self.n, 1)   # shape [batch_size, n, lb]
            tmp_sumF = tmp.sum(dim=1)       # shape [batch_size, lb]
            tmp2 = tmp_sumF * B.view(1, self.lb)  # shape [batch_size, lb]
            val = tmp2.sum(dim=1)                # shape [batch_size]
            return val + bias

        # Feature mean => shape [batch_size, 1]
        feat_mean = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).unsqueeze(1)

        # =============== GATING ===============
        gate_ar_score = self.ar_gate_lin(past_returns)  # [batch_size, 1]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate).unsqueeze(1)
        gate_logits = torch.cat(
            [gate_ar_score, feat_gate_score], dim=1)  # [batch_size, 2]
        gate_weights = F.softmax(gate_logits, dim=1)  # [batch_size, 2]

        # gate_weights[:,0] => AR mixture weight
        # gate_weights[:,1] => FEAT mixture weight

        # =============== COMPUTE MIXTURE'S MEAN ===============
        # For a Normal distribution: E[v] = mu
        # so the mixture mean is a weighted sum of the two means.
        ar_comp_mean = ar_mean.squeeze(-1)     # shape [batch_size]
        feat_comp_mean = feat_mean.squeeze(-1)   # shape [batch_size]

        mixture_mean = (
            gate_weights[:, 0] * ar_comp_mean +
            gate_weights[:, 1] * feat_comp_mean
        )
        return mixture_mean

    def negative_log_likelihood(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        target_vol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mixture negative log-likelihood for the Normal-based mixture model
        plus a penalty for negative means.

        negative_log_likelihood = - sum(log(gate0 * NormalPDF_AR + gate1 * NormalPDF_Feat))
                                  + hinge penalty if means < 0.

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
            target_vol:   [batch_size]
        Returns:
            final_loss: scalar (NLL + penalty)
        """
        # --- AR component ---
        ar_mean = self.ar_mean_lin(past_returns)   # [batch_size, 1]
        ar_logvar = self.ar_logvar_lin(past_returns)  # [batch_size, 1]
        ar_sigma = torch.exp(0.5 * ar_logvar)

        # --- Feature component (bilinear) ---
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            val = tmp2.sum(dim=1)
            return val + bias

        feat_mean = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).unsqueeze(1)
        feat_logvar = bilinear_scalar(
            self.A_logvar, self.B_logvar, features, self.bias_logvar).unsqueeze(1)
        feat_sigma = torch.exp(0.5 * feat_logvar)

        # --- Gating ---
        gate_ar_score = self.ar_gate_lin(past_returns)
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate).unsqueeze(1)
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # [batch_size, 2]

        # --- Normal PDF log-likelihood ---
        # For a Normal(μ,σ^2), log p(x) = -0.5 * log(2π) - log σ - (x - μ)^2 / (2σ^2).
        eps = 1e-12
        x = torch.clamp(target_vol, min=eps)  # avoid exact zero

        ar_mean_squeezed = ar_mean.squeeze(-1)
        ar_sigma_squeezed = ar_sigma.squeeze(-1) + eps
        feat_mean_squeezed = feat_mean.squeeze(-1)
        feat_sigma_squeezed = feat_sigma.squeeze(-1) + eps

        # log pdf for AR normal
        ar_log_pdf = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(ar_sigma_squeezed)
            - 0.5 * ((x - ar_mean_squeezed) / ar_sigma_squeezed)**2
        )

        # log pdf for Feature normal
        feat_log_pdf = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(feat_sigma_squeezed)
            - 0.5 * ((x - feat_mean_squeezed) / feat_sigma_squeezed)**2
        )

        comp0 = torch.log(gate_weights[:, 0] + eps) + ar_log_pdf
        comp1 = torch.log(gate_weights[:, 1] + eps) + feat_log_pdf
        log_mix_pdf = torch.logsumexp(torch.stack(
            [comp0, comp1], dim=1), dim=1)  # [batch_size]
        nll = -log_mix_pdf.mean()  # negative log-likelihood

        # --- PENALTY FOR NEGATIVE MEANS ---
        # The paper uses a hinge-like penalty: penalty = alpha * sum( max(0, delta-mu) )
        # We'll do it per batch item and average.

        penalty_ar = torch.relu(
            self.delta - ar_mean_squeezed)  # shape [batch size]
        penalty_feat = torch.relu(self.delta - feat_mean_squeezed)

        penalty = (penalty_ar + penalty_feat).mean()

        # Weighted by penalty_coef
        total_loss = nll + self.penalty_coef * penalty
        return total_loss


class TemporalMixtureWeibullPredictorBilinear(AbstractPredictor):
    """
    Temporal mixture model with Weibull components:

    - AR component:
        v ~ Weibull(k_ar, lambda_ar)
    - Feature-based component:
        v ~ Weibull(k_feat, lambda_feat)

    Gating function is softmax on two logits:
    - gate_ar_score (linear in past_returns)
    - gate_feat_score (bilinear in features)

    Then we compute mixture's mean via:
       E[v] = gate_ar * (lambda_ar * Gamma(1 + 1/k_ar))
             + gate_feat * (lambda_feat * Gamma(1 + 1/k_feat))

    And train with negative log-likelihood (mixture of two Weibull pdfs).
    """

    def __init__(
        self,
        ar_order: int,  # # past lags
        n: int,         # # feature dims per time slice
        lb: int,        # # time slices in the look-back window
        hidden_dim: int = 16,  # just in case
        eps: float = 1e-6,     # stability offset
    ):
        super().__init__()
        self.ar_order = ar_order
        self.n = n
        self.lb = lb
        self.eps = eps

        # ============ AR COMPONENT (raw_k, raw_lambda) ============

        # Since in Weibull distribution estimating the parameters can be more difficult
        # instead of using simple linear transformation we will apply non linear nn
        # that will later be transformed using softplus

        self.ar_net = nn.Sequential(
            nn.Linear(ar_order, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # outputs raw_k and raw_lambda
        )

        # ============ FEATURE COMPONENT (bilinear) ============

        self.feat_net = nn.Sequential(
            nn.Linear(n * lb, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # outputs raw_k, raw_lambda
        )

        # ============ GATING FUNCTION ============

        self.ar_gate_lin = nn.Linear(ar_order, 1)  # AR gating logit

        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

    def bilinear_scalar(self, A, B, X, bias):
        # X: [batch_size, n, lb]
        # A: [n], B: [lb], bias: [1]
        tmp = X * A.view(1, self.n, 1)         # [batch_size, n, lb]
        tmp_sumF = tmp.sum(dim=1)             # [batch_size, lb]
        tmp2 = tmp_sumF * B.view(1, self.lb)  # [batch_size, lb]
        val = tmp2.sum(dim=1)                 # [batch_size]
        return val + bias

    def _forward(self, past_returns, features, *args, **kwargs):
        """
        Return mixture's *mean* = gate_ar * mean(AR) + gate_feat * mean(Feat),
        where mean(Weibull(k, lambda)) = lambda * Gamma(1 + 1/k).
        """
        batch_size = past_returns.size(0)

        # --- AR shape & scale
        raw_k_ar, raw_lam_ar = self.ar_net(past_returns).chunk(2, dim=-1)
        k_ar = F.softplus(raw_k_ar) + self.eps
        lam_ar = F.softplus(raw_lam_ar) + self.eps

        # --- Feature shape & scale (bilinear)
        features_flat = features.view(
            features.size(0), -1)  # [batch_size, n*lb]
        raw_k_feat, raw_lam_feat = self.feat_net(
            features_flat).chunk(2, dim=-1)
        k_feat = F.softplus(raw_k_feat) + self.eps
        lam_feat = F.softplus(raw_lam_feat) + self.eps

        # --- Gating
        gate_ar_score = self.ar_gate_lin(past_returns)    # [batch_size,1]
        feat_gate_score = self.bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate)
        feat_gate_score = feat_gate_score.unsqueeze(1)    # [batch_size,1]
        gate_logits = torch.cat(
            [gate_ar_score, feat_gate_score], dim=1)  # [batch_size,2]
        gate_weights = F.softmax(gate_logits, dim=1)   # [batch_size,2]

        # means of each component:
        # E[X] = lambda * Gamma(1 + 1/k).
        # We'll use torch.special.gammaln if available, else approximate Gamma...
        # For simplicity, let's do an exponent of logGamma:
        #   Gamma(1 + 1/k) = exp( logGamma(1 + 1/k) ).
        # We'll clamp (1/k) as well.
        inv_k_ar = (1.0 / k_ar).clamp_max(10.0)  # avoid extreme expansions
        inv_k_feat = (1.0 / k_feat).clamp_max(10.0)

        gamma_ar = torch.exp(torch.special.gammaln(1.0 + inv_k_ar))
        gamma_feat = torch.exp(torch.special.gammaln(1.0 + inv_k_feat))

        ar_mean = lam_ar * gamma_ar  # shape [batch_size,1]
        feat_mean = lam_feat * gamma_feat

        # Weighted mixture
        mixture_mean = (
            gate_weights[:, 0:1] * ar_mean +
            gate_weights[:, 1:2] * feat_mean
        ).squeeze(1)  # [batch_size]

        return mixture_mean

    def negative_log_likelihood(self, past_returns, features, target_vol):
        """
        Mixture NLL with Weibull components.

        mixture_pdf = gate_ar * f_weibull_AR + gate_feat * f_weibull_Feat
        log( mixture_pdf ) = logsumexp( log(gate_ar) + log_pdf_AR,
                                        log(gate_feat) + log_pdf_Feat )
        """
        eps = 1e-12
        # [batch_size], must be >0 for Weibull
        x = torch.clamp(target_vol, min=eps)

        # --- AR shape, scale
        raw_k_ar, raw_lam_ar = self.ar_net(past_returns).chunk(2, dim=-1)
        k_ar = F.softplus(raw_k_ar) + self.eps
        lam_ar = F.softplus(raw_lam_ar) + self.eps

        # --- Feat shape, scale
        features_flat = features.view(
            features.size(0), -1)  # [batch_size, n*lb]
        raw_k_feat, raw_lam_feat = self.feat_net(
            features_flat).chunk(2, dim=-1)
        k_feat = F.softplus(raw_k_feat) + self.eps
        lam_feat = F.softplus(raw_lam_feat) + self.eps

        # --- Gating
        gate_ar_score = self.ar_gate_lin(past_returns)
        feat_gate_score = self.bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate).unsqueeze(1)
        gate_logits = torch.cat(
            [gate_ar_score, feat_gate_score], dim=1)  # [batch_size,2]
        gate_weights = F.softmax(gate_logits, dim=1)  # shape [batch_size,2]

        # --- Weibull log-pdf function
        # log f(x | k, lam) = log(k) - log(lam) + (k-1)*log(x/lam) - (x/lam)^k
        # We'll define a small helper:

        def weibull_logpdf(x, k, lam):
            # x: [batch_size]
            # k, lam: [batch_size]
            # watch shapes => k, lam are 2D; we can squeeze for ops
            k_squeezed = k.squeeze(-1)
            lam_squeezed = lam.squeeze(-1)
            log_x_lam = torch.log(x / lam_squeezed)
            return (
                torch.log(k_squeezed + eps)
                - torch.log(lam_squeezed + eps)
                + (k_squeezed - 1.0)*log_x_lam
                - (x / lam_squeezed).pow(k_squeezed)
            )

        ar_log_pdf = weibull_logpdf(x, k_ar, lam_ar)
        feat_log_pdf = weibull_logpdf(x, k_feat, lam_feat)

        # mixture log-likelihood: log(gate0*f0 + gate1*f1)
        comp0 = torch.log(gate_weights[:, 0] + eps) + ar_log_pdf
        comp1 = torch.log(gate_weights[:, 1] + eps) + feat_log_pdf
        stacked = torch.stack([comp0, comp1], dim=1)  # [batch_size, 2]
        log_mix_pdf = torch.logsumexp(stacked, dim=1)  # [batch_size]

        nll = -log_mix_pdf.mean()
        return nll


'''
class TMFixed(AbstractPredictor):
    """
    A temporal mixture model with reverse-gaussian components:

      - AR component (depends on past_returns): 
          vol ~ reverse_gaussian(ar_mean, ar_lambda)
      - Feature component (depends on a bilinear function of features): 
          log(vol) ~ Normal(feat_mean, feat_lambda)

    The gating function is also a mixture of:
      - AR gate logit: linear in past_returns
      - Feature gate logit: bilinear in features

    Then we take a softmax to form gating weights.

    In this class we take lambda parameter to be global (different for feat and ar). 
    It is simpler approach assuming some sort of stationarity / homoskedasticity. It prevents overfitting.
    """

    def __init__(
        self,
        ar_order: int,   # number of past lags used for AR
        n: int,          # number of "feature dims" per time slice
        lb: int,         # how many time slices in the look-back window for features
        # number of parameters to be used as global parameters (params apart from mean)
        n_param: int,
    ):
        """
        Args:
            ar_order: how many past returns/vols are used for AR part
            n:        dimension of each feature slice
            lb:       how many time slices of features we have
        """
        super().__init__()

        self.ar_order = ar_order
        self.n = n
        self.lb = lb
        self.n_param = n_param

        # ============ AR COMPONENT (MEAN & global LAMBDA) ============
        #
        #  ar_mean_lin:  linear => [ar_order] -> [1]
        #  ar_logvar_lin: same, if we want separate variance or log-variance
        #
        self.ar_mean_lin = nn.Linear(ar_order, 1)    # for mu_AR
        self.ar_param = nn.Parameter(torch.tensor(n_param))

        # ============ FEATURE COMPONENT (BILINEAR) ============
        #
        # We'll do a rank-1 bilinear form for the mean:
        #   feat_mean = A_mean^T * X * B_mean
        #   (where X is shape [n, lb])
        #
        # So we only store (A_mean in R^n) and (B_mean in R^lb).
        # Similarly for logvar:
        #   feat_logvar = A_logvar^T * X * B_logvar
        #
        self.A_mean = nn.Parameter(torch.randn(n))
        self.B_mean = nn.Parameter(torch.randn(lb))
        self.bias_mean = nn.Parameter(torch.zeros(1))  # optional bias

        self.feat_param = nn.Parameter(torch.tensor(n_param))

        # ============ GATING FUNCTION ============
        #
        #  We'll produce two logits:
        #   1) AR gate logit => linear in [ar_order]
        #   2) FEAT gate logit => bilinear in X
        #
        # Then do a softmax over them => gating weights in [0,1] that sum to 1
        #
        self.ar_gate_lin = nn.Linear(ar_order, 1)  # AR gating

        # Feature gating (bilinear)
        self.A_gate = nn.Parameter(torch.randn(n))
        self.B_gate = nn.Parameter(torch.randn(lb))
        self.bias_gate = nn.Parameter(torch.zeros(1))

    def _forward(
        self,
        past_returns: torch.Tensor,   # shape [batch_size, ar_order]
        features: torch.Tensor,       # shape [batch_size, n, lb]
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Return the mixture's *mean* forecast E[v_h]. 
        That is  g_AR * E[v_h|AR] + g_FEAT * E[v_h|Feat].
        (where E[v_h] for log-normal is exp(mu + 0.5 sigma^2))

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
        Returns:
            mixture_mean: [batch_size]
        """

        batch_size = past_returns.shape[0]

        # =============== AR COMPONENT ===============
        # Mean of log(vol):  shape [batch_size, 1]
        ar_mean = self.ar_mean_lin(past_returns)

        ar_param = self.ar_param.expand(batch_size, self.n_param)

        # =============== FEATURE COMPONENT (BILINEAR) ===============
        # We'll define a small helper that does A^T X B for each batch item:
        def bilinear_scalar(A, B, X, bias):
            """
            A in R^n
            B in R^lb
            X in R^[batch_size, n, lb]
            bias in R^[1]
            Return => shape [batch_size]
            """
            # multiply along feature dimension:
            # scale each "row" by A (for each batch)
            tmp = X * A.view(1, self.n, 1)
            # sum over features => shape [batch_size, lb]
            tmp_sumF = tmp.sum(dim=1)

            # multiply along time dimension by B
            tmp2 = tmp_sumF * B.view(1, self.lb)  # shape [batch_size, lb]
            # here similiarly we sum over the second dimension, that is here lb => shape [batch_size]
            val = tmp2.sum(dim=1)
            return val + bias  # shape [batch_size]

        # (a) mean of log-vol => shape [batch_size]
        feat_mean = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean)
        # expand dims => shape [batch_size, 1]
        feat_mean = feat_mean.unsqueeze(
            1)  # for further technical reasons

        feat_param = self.feat_param.expand(batch_size, self.n_param)

        # =============== GATING ===============
        # AR gating logit => shape [batch_size, 1]
        gate_ar_score = self.ar_gate_lin(past_returns)

        # Feature gating logit => shape [batch_size]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate)
        feat_gate_score = feat_gate_score.unsqueeze(
            1)  # => shape [batch_size, 1]

        # stack => shape [batch_size, 2], then softmax
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # => [batch_size, 2]

        # gate_weights[:,0] => AR mixture weight
        # gate_weights[:,1] => FEAT mixture weight

        # =============== COMPUTE MIXTURE'S MEAN ===============
        # Here we are using the assumption of reverse gaussian distribution
        # shape => [batch_size]

        # Inverse Gaussian expectation: E[X] = μ

        ar_comp_mean = ar_mean.squeeze(-1)
        feat_comp_mean = feat_mean.squeeze(-1)

        mixture_mean = gate_weights[:, 0] * \
            ar_comp_mean + gate_weights[:, 1] * feat_comp_mean

        return mixture_mean

    def negative_log_likelihood(
        self,
        past_returns: torch.Tensor,
        features: torch.Tensor,
        target_vol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional: mixture log-likelihood for training.
        negative_log_likelihood = - sum(log(gate0 * LNpdf_AR + gate1 * LNpdf_Feat)).
        (LNpdf_*) is the lognormal pdf for each component.

        Args:
            past_returns: [batch_size, ar_order]
            features:     [batch_size, n, lb]
            target_vol:   [batch_size]
        Returns:
            nll: scalar
        """
        # The logic up until "Mixture logpdf" is the same as in forward

        batch_size = past_returns.shape[0]

        # --- AR component params ---
        ar_mean = self.ar_mean_lin(past_returns)  # [batch_size, 1]

        lambda_ar = self.lambda_ar.expand(batch_size, 1)

        # --- Feature component params (bilinear) ---
        def bilinear_scalar(A, B, X, bias):
            tmp = X * A.view(1, self.n, 1)
            tmp_sumF = tmp.sum(dim=1)
            tmp2 = tmp_sumF * B.view(1, self.lb)
            val = tmp2.sum(dim=1)
            return val + bias

        def inverse_gaussian_logpdf(x, mu, lam, eps=1e-12):
            x = torch.clamp(x, min=eps)
            term1 = 0.5 * torch.log(lam / (2 * torch.pi * x**3))
            term2 = -(lam * (x - mu)**2) / (2 * mu**2 * x)
            return term1 + term2

        feat_mean = bilinear_scalar(
            self.A_mean, self.B_mean, features, self.bias_mean).unsqueeze(1)

        lambda_feat = self.lambda_feat.expand(batch_size, 1)

        # --- Gating function ---
        gate_ar_score = self.ar_gate_lin(past_returns)  # [batch_size, 1]
        feat_gate_score = bilinear_scalar(
            self.A_gate, self.B_gate, features, self.bias_gate).unsqueeze(1)
        gate_logits = torch.cat([gate_ar_score, feat_gate_score], dim=1)
        gate_weights = F.softmax(gate_logits, dim=1)  # [batch_size, 2]

        # --- Mixture logpdf ---
        # log-inverse_gaussian pdf:
        eps = 1e-12
        # we are bounding x from below to deal with indefinitness of log at x=0
        x = torch.clamp(target_vol, min=eps)
        logx = torch.log(x)

        ar_log_pdf = inverse_gaussian_logpdf(x, ar_mean, lambda_ar)
        feat_log_pdf = inverse_gaussian_logpdf(x, feat_mean, lambda_feat)

        comp0 = torch.log(gate_weights[:, 0] + eps) + ar_log_pdf
        comp1 = torch.log(gate_weights[:, 1] + eps) + feat_log_pdf
        # log mixture = logsumexp( comp0, comp1 )
        log_mix_pdf = torch.logsumexp(
            torch.stack([comp0, comp1], dim=1), dim=1)

        nll = - log_mix_pdf.mean()
        return nll
'''
