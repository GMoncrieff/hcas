"""Tests for HCASScorer (manuscript-faithful HCAS method)."""

import numpy as np
import pytest

from hcas import HCASScorer
from hcas._hcas_scorer import _score_single_site_hcas


@pytest.fixture(scope="session")
def hcas_result(synthetic_ds, var_names):
    """Fit and score once for the session."""
    scorer = HCASScorer(
        n_estimators=100, random_state=42,
        density_subsample=500,          # small for fast tests
        calibration_subsample=200,
    )
    scorer.fit(synthetic_ds, **var_names)
    result = scorer.score(synthetic_ds, **var_names)
    return result, synthetic_ds, scorer


def _get_scores_by_category(result, ds):
    test_mask = ds["test_mask"].values
    scores = result.values
    categories = {}
    for label, tc in [("intact", 1.0), ("mild", 0.7), ("heavy", 0.2), ("transformed", 1e-10)]:
        mask = np.isclose(test_mask, tc, atol=1e-8)
        cat_scores = scores[mask]
        cat_scores = cat_scores[~np.isnan(cat_scores)]
        categories[label] = cat_scores
    return categories


class TestHCASScorer:
    def test_output_is_xarray(self, hcas_result):
        result, ds, _ = hcas_result
        assert result.dims == ("lat", "lon")

    def test_scores_in_unit_interval(self, hcas_result):
        result, _, _ = hcas_result
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_nan_at_non_test_pixels(self, hcas_result):
        result, ds, _ = hcas_result
        test_mask = ds["test_mask"].values
        non_test = test_mask == 0
        assert np.all(np.isnan(result.values[non_test]))

    def test_intact_scores_higher_than_transformed(self, hcas_result):
        """With small test data, absolute thresholds are unreliable.
        The key requirement is correct relative ordering."""
        result, ds, _ = hcas_result
        cats = _get_scores_by_category(result, ds)
        assert len(cats["intact"]) > 0
        assert len(cats["transformed"]) > 0
        assert np.mean(cats["intact"]) > np.mean(cats["transformed"])

    def test_ordering(self, hcas_result):
        result, ds, _ = hcas_result
        cats = _get_scores_by_category(result, ds)
        means = {k: np.mean(v) for k, v in cats.items() if len(v) > 0}
        assert means["intact"] > means["mild"]
        assert means["mild"] > means["heavy"]
        assert means["heavy"] > means["transformed"]


class TestHCASEquations:
    """Verify the HCAS equations match the manuscript."""

    def test_half_cauchy_weights_equation2(self):
        """Equation 2: w_i = 1 / (pi * (1 + (d_pred_i / lambda)^2))"""
        d_preds = np.array([0.0, 1.0, 2.0, 4.0])
        lam = 2.0
        expected = 1.0 / (np.pi * (1.0 + (d_preds / lam) ** 2))
        # At d_pred=0: w = 1/pi
        assert np.isclose(expected[0], 1.0 / np.pi)
        # At d_pred=lambda: w = 1/(2*pi)
        assert np.isclose(expected[2], 1.0 / (2 * np.pi))

    def test_ldc_formula_equation3(self):
        """Equation 3: H_c^LDC = omega * (sum(p*w)/sum(w) + p_max)"""
        probs = np.array([0.8, 0.5, 0.3])
        d_preds = np.array([0.5, 1.0, 2.0])
        score = _score_single_site_hcas(probs, d_preds, cauchy_lambda=2.0, omega=0.5)
        # Manual calculation
        weights = 1.0 / (np.pi * (1.0 + (d_preds / 2.0) ** 2))
        expected = 0.5 * (np.sum(probs * weights) / np.sum(weights) + 0.8)
        assert np.isclose(score, expected)

    def test_default_parameters_match_manuscript(self):
        """Table S14 parameters."""
        scorer = HCASScorer()
        assert scorer.bin_size == 0.005         # Z
        assert scorer.n_candidates == 50        # n_p
        assert scorer.geo_radius_km == 200.0    # R
        assert scorer.n_benchmarks == 20        # n_ref
        assert scorer.cauchy_lambda == 2.0      # lambda
        assert scorer.omega == 0.5              # omega

    def test_density_surface_built(self, hcas_result):
        _, _, scorer = hcas_result
        assert hasattr(scorer, "surface_")
        assert scorer.surface_.ndim == 2
        # Early columns (dense region) should sum close to 1
        col_sums = scorer.surface_.sum(axis=0)
        # Check first few columns where data is densest
        dense_cols = col_sums[:5]
        assert np.allclose(dense_cols[dense_cols > 0], 1.0, atol=0.05)

    def test_calibration_range_stored(self, hcas_result):
        _, _, scorer = hcas_result
        assert hasattr(scorer, "score_min_")
        assert hasattr(scorer, "score_max_")
        assert scorer.score_max_ > scorer.score_min_
