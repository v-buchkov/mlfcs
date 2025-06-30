import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl
from torch.distributions import Weibull


class InverseGaussian(Distribution):
    arg_constraints = {'loc': constraints.positive,
                       'concentration': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        mu = self.loc.expand(shape)
        lam = self.concentration.expand(shape)

        v = torch.randn(shape, device=mu.device)
        y = v ** 2
        x = mu + (mu ** 2 * y) / (2 * lam) - (mu / (2 * lam)) * \
            torch.sqrt(4 * mu * lam * y + mu ** 2 * y ** 2)
        z = torch.rand(shape, device=mu.device)
        return torch.where(z <= mu / (mu + x), x, (mu ** 2) / x)

    def rsample(self, sample_shape=torch.Size()):
        # not truly reparameterizable – fallback to .sample()
        return self.sample(sample_shape)

    def log_prob(self, value):
        mu = self.loc
        lam = self.concentration
        return 0.5 * torch.log(lam / (2 * torch.pi * value ** 3)) - (lam * (value - mu) ** 2) / (2 * mu ** 2 * value)

    def entropy(self):
        raise NotImplementedError(
            "Entropy not implemented for InverseGaussian")

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (self.loc ** 3) / self.concentration


@register_kl(InverseGaussian, InverseGaussian)
def _kl_inverse_gaussian(p, q):
    mu_p, lam_p = p.loc, p.concentration
    mu_q, lam_q = q.loc, q.concentration
    term1 = torch.log(lam_q / lam_p)
    term2 = (lam_p / lam_q) * ((mu_p - mu_q) ** 2) / mu_q ** 2
    term3 = (lam_p / lam_q) - 1
    return 0.5 * (term1 + term2 - term3)


@register_kl(Weibull, Weibull)
def _kl_weibull_weibull(p, q):
    k1, l1 = p.concentration, p.scale
    k2, l2 = q.concentration, q.scale

    # Przybliżony analityczny wzór na KL(Weibull || Weibull)
    t1 = torch.log(k1 / k2)
    t2 = (torch.special.gammaln(1 + 1 / k1) -
          torch.special.gammaln(1 + 1 / k2))
    t3 = (l1 / l2).pow(k2)
    t4 = (k1 / k2 - 1) * 0.57721566490153286060651209008240243

    return t1 + t2 + t3 + t4 - 1
