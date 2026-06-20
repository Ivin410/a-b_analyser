"""
ab_stats.py
Core statistical engine for the A/B Test Analyzer.

Supports two common test types:
1. Conversion rate tests (binary outcome: converted / not converted)
   -> Two-proportion z-test
2. Continuous metric tests (e.g. revenue, time-on-page, order value)
   -> Welch's t-test (does not assume equal variances)

All functions return plain Python dicts so they're easy to render
in Streamlit or convert to JSON.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy import stats


@dataclass
class ConversionResult:
    control_n: int
    control_conversions: int
    control_rate: float
    variant_n: int
    variant_conversions: int
    variant_rate: float
    absolute_uplift: float
    relative_uplift_pct: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    alpha: float
    power: float
    recommendation: str


@dataclass
class ContinuousResult:
    control_n: int
    control_mean: float
    control_std: float
    variant_n: int
    variant_mean: float
    variant_std: float
    absolute_uplift: float
    relative_uplift_pct: float
    t_score: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    alpha: float
    cohens_d: float
    recommendation: str


def analyze_conversion(
    control_n: int,
    control_conversions: int,
    variant_n: int,
    variant_conversions: int,
    alpha: float = 0.05,
) -> ConversionResult:
    """
    Two-proportion z-test for conversion rate experiments.
    """
    if control_n <= 0 or variant_n <= 0:
        raise ValueError("Sample sizes must be greater than 0.")
    if control_conversions > control_n or variant_conversions > variant_n:
        raise ValueError("Conversions cannot exceed sample size.")

    p_control = control_conversions / control_n
    p_variant = variant_conversions / variant_n

    # Pooled proportion for the standard error under H0
    p_pool = (control_conversions + variant_conversions) / (control_n + variant_n)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / variant_n))

    if se_pool == 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = float((p_variant - p_control) / se_pool)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    # Confidence interval on the difference (unpooled SE, standard approach)
    se_diff = math.sqrt(
        (p_control * (1 - p_control) / control_n)
        + (p_variant * (1 - p_variant) / variant_n)
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_variant - p_control
    ci_low = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    significant = bool(p_value < alpha)

    relative_uplift = ((p_variant - p_control) / p_control * 100) if p_control > 0 else float("nan")

    power = _power_two_proportions(p_control, p_variant, control_n, variant_n, alpha)

    recommendation = _recommendation_text(significant, relative_uplift, power)

    return ConversionResult(
        control_n=control_n,
        control_conversions=control_conversions,
        control_rate=p_control,
        variant_n=variant_n,
        variant_conversions=variant_conversions,
        variant_rate=p_variant,
        absolute_uplift=diff,
        relative_uplift_pct=relative_uplift,
        z_score=z_score,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        alpha=alpha,
        power=power,
        recommendation=recommendation,
    )


def analyze_continuous(
    control_values: np.ndarray,
    variant_values: np.ndarray,
    alpha: float = 0.05,
) -> ContinuousResult:
    """
    Welch's t-test for continuous metrics (revenue, session length, etc).
    Does not assume equal variances between groups.
    """
    control_values = np.asarray(control_values, dtype=float)
    variant_values = np.asarray(variant_values, dtype=float)

    control_values = control_values[~np.isnan(control_values)]
    variant_values = variant_values[~np.isnan(variant_values)]

    if len(control_values) < 2 or len(variant_values) < 2:
        raise ValueError("Each group needs at least 2 data points.")

    mean_c, mean_v = control_values.mean(), variant_values.mean()
    std_c, std_v = control_values.std(ddof=1), variant_values.std(ddof=1)
    n_c, n_v = len(control_values), len(variant_values)

    t_score, p_value = stats.ttest_ind(variant_values, control_values, equal_var=False)
    t_score, p_value = float(t_score), float(p_value)

    # Welch-Satterthwaite degrees of freedom for the CI
    se_c2 = std_c**2 / n_c
    se_v2 = std_v**2 / n_v
    se_diff = math.sqrt(se_c2 + se_v2)
    df = (se_c2 + se_v2) ** 2 / ((se_c2**2 / (n_c - 1)) + (se_v2**2 / (n_v - 1)))

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean_v - mean_c
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    significant = bool(p_value < alpha)
    relative_uplift = (diff / mean_c * 100) if mean_c != 0 else float("nan")

    # Cohen's d (pooled std) as an effect size indicator
    pooled_std = math.sqrt(((n_c - 1) * std_c**2 + (n_v - 1) * std_v**2) / (n_c + n_v - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

    recommendation = _recommendation_text(significant, relative_uplift, None, cohens_d=cohens_d)

    return ContinuousResult(
        control_n=n_c,
        control_mean=mean_c,
        control_std=std_c,
        variant_n=n_v,
        variant_mean=mean_v,
        variant_std=std_v,
        absolute_uplift=diff,
        relative_uplift_pct=relative_uplift,
        t_score=t_score,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        alpha=alpha,
        cohens_d=cohens_d,
        recommendation=recommendation,
    )


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect_pct: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Sample size needed PER GROUP to detect a relative MDE on a baseline
    conversion rate, using a standard two-proportion z-test power formula.
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1.")

    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect_pct / 100)
    p2 = min(p2, 0.999999)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    pooled = (p1 + p2) / 2
    n = (
        (z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p2 - p1) ** 2
    )
    return int(math.ceil(n))


def _power_two_proportions(p1, p2, n1, n2, alpha):
    """Post-hoc statistical power for a two-proportion z-test."""
    if p1 == p2:
        return alpha  # no real effect; power collapses to alpha
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se1 = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se0 == 0 or se1 == 0:
        return 0.0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = (abs(p2 - p1) - z_alpha * se0) / se1
    return float(stats.norm.cdf(z))


def _recommendation_text(significant, relative_uplift, power, cohens_d=None):
    if significant and relative_uplift is not None and relative_uplift > 0:
        msg = "✅ Variant shows a statistically significant improvement. Consider shipping it."
    elif significant and relative_uplift is not None and relative_uplift < 0:
        msg = "⚠️ Variant is statistically significantly WORSE than control. Do not ship."
    else:
        msg = "ℹ️ No statistically significant difference detected. "
        if power is not None and power < 0.8:
            msg += f"Statistical power is low ({power:.0%}) — consider collecting more data before concluding."
        else:
            msg += "Consider running longer or checking if the effect size is too small to matter."
    return msg
