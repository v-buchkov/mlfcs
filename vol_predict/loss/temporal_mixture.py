import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
from abc import ABC
from __future__ import annotations

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

        lambda_ar = self.lambda_ar.expand(batch_size, 1)

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

        lambda_feat = self.lambda_feat.expand(batch_size, 1)

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
        n_param: int,    # number of parameters to be used as global parameters (params apart from mean)
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

        lambda_ar = self.lambda_ar.expand(batch_size, 1)

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

        lambda_feat = self.lambda_feat.expand(batch_size, 1)

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
