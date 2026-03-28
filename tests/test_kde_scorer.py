"""Tests for RFProximityScorer (RF proximity-weighted KDE method)."""

import numpy as np
import pytest

from hcas import RFProximityScorer


@pytest.fixture(scope="session")
def kde_result(synthetic_ds, var_names):
    """Fit and score once for the session."""
    scorer = RFProximityScorer(
        n_estimators=100, random_state=42, min_samples_leaf=5,
    )
    scorer.fit(synthetic_ds, **var_names)
    result = scorer.score(synthetic_ds, **var_names)
    return result, synthetic_ds


def _get_scores_by_category(result, ds):
    """Extract scores grouped by true condition value."""
    test_mask = ds["test_mask"].values
    scores = result.values
    categories = {}
    # true condition values: 1.0 (intact), 0.7 (mild), 0.2 (heavy), ~0 (transformed)
    for label, tc in [("intact", 1.0), ("mild", 0.7), ("heavy", 0.2), ("transformed", 1e-10)]:
        mask = np.isclose(test_mask, tc, atol=1e-8)
        cat_scores = scores[mask]
        cat_scores = cat_scores[~np.isnan(cat_scores)]
        categories[label] = cat_scores
    return categories


class TestRFProximityScorer:
    def test_output_is_xarray(self, kde_result):
        result, ds = kde_result
        assert result.dims == ("lat", "lon")
        assert "lat" in result.coords
        assert "lon" in result.coords

    def test_scores_in_unit_interval(self, kde_result):
        result, _ = kde_result
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_nan_at_non_test_pixels(self, kde_result):
        result, ds = kde_result
        test_mask = ds["test_mask"].values
        non_test = test_mask == 0
        assert np.all(np.isnan(result.values[non_test]))

    def test_intact_scores_high(self, kde_result):
        result, ds = kde_result
        cats = _get_scores_by_category(result, ds)
        assert len(cats["intact"]) > 0
        assert np.mean(cats["intact"]) > 0.6

    def test_transformed_scores_low(self, kde_result):
        result, ds = kde_result
        cats = _get_scores_by_category(result, ds)
        assert len(cats["transformed"]) > 0
        assert np.mean(cats["transformed"]) < 0.4

    def test_ordering(self, kde_result):
        result, ds = kde_result
        cats = _get_scores_by_category(result, ds)
        means = {k: np.mean(v) for k, v in cats.items() if len(v) > 0}
        assert means["intact"] > means["mild"]
        assert means["mild"] > means["heavy"]
        assert means["heavy"] > means["transformed"]

    def test_custom_parameters(self, synthetic_ds, var_names):
        scorer = RFProximityScorer(
            n_estimators=50, kde_bw_factor=1.5, random_state=99,
        )
        scorer.fit(synthetic_ds, **var_names)
        assert scorer.n_estimators == 50
        assert scorer.kde_bw_factor == 1.5
