"""
bayesian_stats.py
Bayesian A/B testing engine.

Conversion-rate tests use the Beta-Binomial conjugate model, which has a
closed-form posterior — no MCMC needed, fast and exact.

Continuous-metric tests use Monte Carlo simulation with weakly-informative
priors on the mean and variance of each group (Normal-Inverse-Gamma-style
approximation via bootstrap-of-the-posterior, which is simple, dependency-light,
and good enough for practical decision-making without requiring PyMC/Stan).

All results are framed in plain-language probability statements, which is the
main advantage of the Bayesian approach over p-values: "there's an 87% chance
B beats A" instead of "p = 0.03, reject the null."
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class BayesianConversionResult:
    control_n: int
    control_conversions: int
    variant_n: int
    variant_conversions: int
    control_rate: float
    variant_rate: float
    prob_variant_better: float
    expected_uplift_pct: float
    credible_interval_uplift: tuple  # (low, high) for relative uplift %
    control_samples: np.ndarray
    variant_samples: np.ndarray
    recommendation: str


@dataclass
class BayesianContinuousResult:
    control_n: int
    variant_n: int
    control_mean: float
    variant_mean: float
    prob_variant_better: float
    expected_uplift_pct: float
    credible_interval_uplift: tuple
    control_samples: np.ndarray
    variant_samples: np.ndarray
    recommendation: str


def bayesian_conversion_analysis(
    control_n: int,
    control_conversions: int,
    variant_n: int,
    variant_conversions: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    credible_level: float = 0.95,
    seed: int = 42,
) -> BayesianConversionResult:
    """
    Beta-Binomial conjugate model.

    prior_alpha / prior_beta = 1, 1 gives a flat/uninformative prior
    (uniform over [0,1] conversion rate). Increase both proportionally
    to encode a stronger prior belief (e.g. alpha=10, beta=90 says
    "I expect ~10% conversion before seeing any data").
    """
    if control_n <= 0 or variant_n <= 0:
        raise ValueError("Sample sizes must be greater than 0.")
    if control_conversions > control_n or variant_conversions > variant_n:
        raise ValueError("Conversions cannot exceed sample size.")

    rng = np.random.default_rng(seed)

    # Posterior is Beta(prior_alpha + successes, prior_beta + failures)
    post_alpha_c = prior_alpha + control_conversions
    post_beta_c = prior_beta + (control_n - control_conversions)
    post_alpha_v = prior_alpha + variant_conversions
    post_beta_v = prior_beta + (variant_n - variant_conversions)

    control_samples = rng.beta(post_alpha_c, post_beta_c, n_samples)
    variant_samples = rng.beta(post_alpha_v, post_beta_v, n_samples)

    prob_variant_better = float(np.mean(variant_samples > control_samples))

    relative_uplift_samples = (variant_samples - control_samples) / control_samples * 100
    expected_uplift = float(np.mean(relative_uplift_samples))

    lower_pct = (1 - credible_level) / 2 * 100
    upper_pct = 100 - lower_pct
    ci = (
        float(np.percentile(relative_uplift_samples, lower_pct)),
        float(np.percentile(relative_uplift_samples, upper_pct)),
    )

    recommendation = _bayesian_recommendation(prob_variant_better, expected_uplift)

    return BayesianConversionResult(
        control_n=control_n,
        control_conversions=control_conversions,
        variant_n=variant_n,
        variant_conversions=variant_conversions,
        control_rate=control_conversions / control_n,
        variant_rate=variant_conversions / variant_n,
        prob_variant_better=prob_variant_better,
        expected_uplift_pct=expected_uplift,
        credible_interval_uplift=ci,
        control_samples=control_samples,
        variant_samples=variant_samples,
        recommendation=recommendation,
    )


def bayesian_continuous_analysis(
    control_values: np.ndarray,
    variant_values: np.ndarray,
    n_samples: int = 100_000,
    credible_level: float = 0.95,
    seed: int = 42,
) -> BayesianContinuousResult:
    """
    Bayesian estimation for continuous metrics via the Bayesian bootstrap.

    Rather than assuming a parametric likelihood (Normal, etc.), this
    resamples each observed dataset with Dirichlet-distributed weights,
    which approximates the posterior over the mean without strong
    distributional assumptions. This is a standard, lightweight technique
    (Rubin, 1981) that avoids needing PyMC/Stan as a dependency while still
    producing genuine posterior-style uncertainty.
    """
    control_values = np.asarray(control_values, dtype=float)
    variant_values = np.asarray(variant_values, dtype=float)
    control_values = control_values[~np.isnan(control_values)]
    variant_values = variant_values[~np.isnan(variant_values)]

    if len(control_values) < 2 or len(variant_values) < 2:
        raise ValueError("Each group needs at least 2 data points.")

    rng = np.random.default_rng(seed)

    def bayesian_bootstrap_means(data, n_iter):
        n = len(data)
        weights = rng.dirichlet(np.ones(n), size=n_iter)
        return weights @ data

    control_samples = bayesian_bootstrap_means(control_values, n_samples)
    variant_samples = bayesian_bootstrap_means(variant_values, n_samples)

    prob_variant_better = float(np.mean(variant_samples > control_samples))

    relative_uplift_samples = (variant_samples - control_samples) / np.abs(control_samples) * 100
    expected_uplift = float(np.mean(relative_uplift_samples))

    lower_pct = (1 - credible_level) / 2 * 100
    upper_pct = 100 - lower_pct
    ci = (
        float(np.percentile(relative_uplift_samples, lower_pct)),
        float(np.percentile(relative_uplift_samples, upper_pct)),
    )

    recommendation = _bayesian_recommendation(prob_variant_better, expected_uplift)

    return BayesianContinuousResult(
        control_n=len(control_values),
        variant_n=len(variant_values),
        control_mean=float(control_values.mean()),
        variant_mean=float(variant_values.mean()),
        prob_variant_better=prob_variant_better,
        expected_uplift_pct=expected_uplift,
        credible_interval_uplift=ci,
        control_samples=control_samples,
        variant_samples=variant_samples,
        recommendation=recommendation,
    )


def _bayesian_recommendation(prob_variant_better: float, expected_uplift: float) -> str:
    if prob_variant_better >= 0.95:
        return (f"✅ Strong evidence for the variant — {prob_variant_better:.1%} probability it beats "
                f"control (expected uplift {expected_uplift:+.1f}%). Reasonable to ship.")
    elif prob_variant_better <= 0.05:
        return (f"⚠️ Strong evidence AGAINST the variant — only {prob_variant_better:.1%} probability "
                f"it beats control. Do not ship.")
    elif prob_variant_better >= 0.80:
        return (f"🟡 Variant is favored ({prob_variant_better:.1%} probability of beating control) but "
                f"not yet decisive. Consider more data if the decision is high-stakes.")
    elif prob_variant_better <= 0.20:
        return (f"🟡 Control is favored ({1 - prob_variant_better:.1%} probability of beating variant) "
                f"but not yet decisive. Consider more data if the decision is high-stakes.")
    else:
        return (f"ℹ️ Too close to call — {prob_variant_better:.1%} probability variant beats control. "
                f"Collect more data before deciding.")
