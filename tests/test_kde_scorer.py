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


class TestCalibration:
    """Tests for the optional isotonic calibration step."""

    @pytest.fixture()
    def fitted_scorer(self, synthetic_ds, var_names):
        scorer = RFProximityScorer(
            n_estimators=100, random_state=42, min_samples_leaf=5,
        )
        scorer.fit(synthetic_ds, **var_names)
        return scorer

    @pytest.fixture()
    def raw_scores_and_truth(self, fitted_scorer, synthetic_ds, var_names):
        result = fitted_scorer.score(synthetic_ds, **var_names)
        test_mask = synthetic_ds["test_mask"].values
        mask = test_mask != 0
        raw = result.values[mask]
        true_cond = test_mask[mask]
        return raw, true_cond

    def test_calibrate_returns_self(self, fitted_scorer, raw_scores_and_truth):
        raw, true_cond = raw_scores_and_truth
        ret = fitted_scorer.calibrate(raw, true_cond)
        assert ret is fitted_scorer

    def test_calibrated_scores_in_unit_interval(
        self, fitted_scorer, synthetic_ds, var_names, raw_scores_and_truth
    ):
        raw, true_cond = raw_scores_and_truth
        fitted_scorer.calibrate(raw, true_cond)
        result = fitted_scorer.score(synthetic_ds, **var_names)
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_ordering_preserved_after_calibration(
        self, fitted_scorer, synthetic_ds, var_names, raw_scores_and_truth
    ):
        raw, true_cond = raw_scores_and_truth
        fitted_scorer.calibrate(raw, true_cond)
        result = fitted_scorer.score(synthetic_ds, **var_names)
        cats = _get_scores_by_category(result, synthetic_ds)
        means = {k: np.mean(v) for k, v in cats.items() if len(v) > 0}
        assert means["intact"] > means["mild"]
        assert means["mild"] > means["heavy"]
        assert means["heavy"] > means["transformed"]

    def test_calibration_reduces_error(
        self, fitted_scorer, synthetic_ds, var_names, raw_scores_and_truth
    ):
        raw, true_cond = raw_scores_and_truth
        error_before = np.mean((raw - true_cond) ** 2)
        fitted_scorer.calibrate(raw, true_cond)
        result = fitted_scorer.score(synthetic_ds, **var_names)
        test_mask = synthetic_ds["test_mask"].values
        cal_scores = result.values[test_mask != 0]
        error_after = np.mean((cal_scores - true_cond) ** 2)
        assert error_after <= error_before

    def test_no_calibration_by_default(self, fitted_scorer):
        assert fitted_scorer.calibration_ is None

    def test_remove_calibration(
        self, fitted_scorer, synthetic_ds, var_names, raw_scores_and_truth
    ):
        raw, true_cond = raw_scores_and_truth
        fitted_scorer.calibrate(raw, true_cond)
        assert fitted_scorer.calibration_ is not None
        fitted_scorer.calibration_ = None
        result = fitted_scorer.score(synthetic_ds, **var_names)
        # Should be back to uncalibrated scores
        test_mask = synthetic_ds["test_mask"].values
        scores = result.values[test_mask != 0]
        np.testing.assert_allclose(scores, raw, atol=1e-10)

    def test_calibrate_handles_nans(self, fitted_scorer):
        raw = np.array([0.1, np.nan, 0.5, 0.9])
        true_cond = np.array([0.2, 0.4, np.nan, 0.8])
        fitted_scorer.calibrate(raw, true_cond)
        # Should fit on the 2 valid pairs (0.1→0.2, 0.9→0.8)
        assert fitted_scorer.calibration_ is not None
