"""
tests/test_ab_stats.py
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ab_stats import analyze_conversion, analyze_continuous, required_sample_size


class TestConversionAnalysis:
    def test_clear_winner_is_significant(self):
        # Variant has obviously higher conversion with large sample
        result = analyze_conversion(
            control_n=5000, control_conversions=500,   # 10%
            variant_n=5000, variant_conversions=650,   # 13%
        )
        assert result.significant is True
        assert result.variant_rate > result.control_rate
        assert result.p_value < 0.05

    def test_no_difference_not_significant(self):
        result = analyze_conversion(
            control_n=1000, control_conversions=100,
            variant_n=1000, variant_conversions=101,
        )
        assert result.significant is False

    def test_identical_rates_p_value_near_one(self):
        result = analyze_conversion(1000, 200, 1000, 200)
        assert result.p_value > 0.9

    def test_invalid_conversions_raises(self):
        with pytest.raises(ValueError):
            analyze_conversion(100, 150, 100, 50)  # conversions > n

    def test_zero_sample_size_raises(self):
        with pytest.raises(ValueError):
            analyze_conversion(0, 0, 100, 10)

    def test_relative_uplift_sign(self):
        result = analyze_conversion(1000, 100, 1000, 150)
        assert result.relative_uplift_pct == pytest.approx(50.0, rel=0.01)


class TestContinuousAnalysis:
    def test_clear_difference_is_significant(self):
        np.random.seed(0)
        control = np.random.normal(10, 1, 200)
        variant = np.random.normal(12, 1, 200)
        result = analyze_continuous(control, variant)
        assert result.significant is True
        assert result.variant_mean > result.control_mean

    def test_identical_distributions_not_significant(self):
        np.random.seed(1)
        control = np.random.normal(10, 2, 500)
        variant = np.random.normal(10, 2, 500)
        result = analyze_continuous(control, variant)
        # Not guaranteed but extremely likely with this seed/sample size
        assert result.p_value > 0.01

    def test_handles_nans(self):
        control = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        variant = np.array([2.0, 3.0, 4.0, np.nan, 5.0])
        result = analyze_continuous(control, variant)
        assert result.control_n == 4
        assert result.variant_n == 4

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            analyze_continuous(np.array([1.0]), np.array([1.0, 2.0, 3.0]))

    def test_cohens_d_zero_when_identical(self):
        vals = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 6.0])
        result = analyze_continuous(vals, vals)
        assert result.cohens_d == pytest.approx(0.0, abs=1e-9)


class TestSampleSizeCalculator:
    def test_returns_positive_integer(self):
        n = required_sample_size(baseline_rate=0.10, minimum_detectable_effect_pct=10)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_mde_requires_more_sample(self):
        n_small_effect = required_sample_size(0.10, 5)
        n_large_effect = required_sample_size(0.10, 20)
        assert n_small_effect > n_large_effect

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            required_sample_size(1.5, 10)
        with pytest.raises(ValueError):
            required_sample_size(0, 10)
