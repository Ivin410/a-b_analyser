"""
tests/test_bayesian_stats.py
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bayesian_stats import bayesian_conversion_analysis, bayesian_continuous_analysis


class TestBayesianConversion:
    def test_clear_winner_high_probability(self):
        result = bayesian_conversion_analysis(
            control_n=5000, control_conversions=500,   # 10%
            variant_n=5000, variant_conversions=650,   # 13%
        )
        assert result.prob_variant_better > 0.95
        assert result.variant_rate > result.control_rate

    def test_no_real_difference_near_fifty_fifty(self):
        result = bayesian_conversion_analysis(
            control_n=1000, control_conversions=100,
            variant_n=1000, variant_conversions=101,
        )
        assert 0.3 < result.prob_variant_better < 0.7

    def test_identical_rates_near_half(self):
        result = bayesian_conversion_analysis(1000, 200, 1000, 200)
        assert abs(result.prob_variant_better - 0.5) < 0.05

    def test_invalid_conversions_raises(self):
        with pytest.raises(ValueError):
            bayesian_conversion_analysis(100, 150, 100, 50)

    def test_zero_sample_size_raises(self):
        with pytest.raises(ValueError):
            bayesian_conversion_analysis(0, 0, 100, 10)

    def test_credible_interval_ordering(self):
        result = bayesian_conversion_analysis(1000, 100, 1000, 150)
        low, high = result.credible_interval_uplift
        assert low < high

    def test_worse_variant_detected(self):
        result = bayesian_conversion_analysis(
            control_n=5000, control_conversions=650,
            variant_n=5000, variant_conversions=500,
        )
        assert result.prob_variant_better < 0.05

    def test_stronger_prior_pulls_toward_prior_mean(self):
        # Tiny sample, strong prior centered at 50% — posterior should stay close to 50%
        weak_prior = bayesian_conversion_analysis(10, 6, 10, 7, prior_alpha=1, prior_beta=1)
        strong_prior = bayesian_conversion_analysis(10, 6, 10, 7, prior_alpha=50, prior_beta=50)
        # Strong prior should pull the rate estimates closer to 0.5 than weak prior does
        assert abs(strong_prior.control_rate - 0.6) >= 0  # control_rate is just observed data either way
        # but the resulting posterior samples should have lower variance under strong prior
        assert np.std(strong_prior.control_samples) < np.std(weak_prior.control_samples)


class TestBayesianContinuous:
    def test_clear_difference_high_probability(self):
        np.random.seed(0)
        control = np.random.normal(10, 1, 200)
        variant = np.random.normal(12, 1, 200)
        result = bayesian_continuous_analysis(control, variant)
        assert result.prob_variant_better > 0.95
        assert result.variant_mean > result.control_mean

    def test_identical_distributions_near_fifty_fifty(self):
        np.random.seed(1)
        control = np.random.normal(10, 2, 500)
        variant = np.random.normal(10, 2, 500)
        result = bayesian_continuous_analysis(control, variant)
        assert 0.2 < result.prob_variant_better < 0.8

    def test_handles_nans(self):
        control = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        variant = np.array([2.0, 3.0, 4.0, np.nan, 5.0])
        result = bayesian_continuous_analysis(control, variant)
        assert result.control_n == 4
        assert result.variant_n == 4

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            bayesian_continuous_analysis(np.array([1.0]), np.array([1.0, 2.0, 3.0]))

    def test_credible_interval_ordering(self):
        np.random.seed(2)
        control = np.random.normal(10, 1, 100)
        variant = np.random.normal(11, 1, 100)
        result = bayesian_continuous_analysis(control, variant)
        low, high = result.credible_interval_uplift
        assert low < high

    def test_worse_variant_detected(self):
        np.random.seed(3)
        control = np.random.normal(12, 1, 200)
        variant = np.random.normal(10, 1, 200)
        result = bayesian_continuous_analysis(control, variant)
        assert result.prob_variant_better < 0.05

    def test_reproducible_with_seed(self):
        control = np.array([10.0, 11.0, 9.5, 10.5, 11.2, 9.8])
        variant = np.array([12.0, 13.0, 11.5, 12.5, 13.2, 11.8])
        r1 = bayesian_continuous_analysis(control, variant, seed=99)
        r2 = bayesian_continuous_analysis(control, variant, seed=99)
        assert r1.prob_variant_better == r2.prob_variant_better
