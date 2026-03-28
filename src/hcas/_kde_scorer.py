"""RF proximity-weighted marginal KDE condition scorer.

For each test site, uses Random Forest leaf-based proximity weights to define
a neighbourhood of reference sites, fits weighted 1D marginal KDEs (one per
outcome variable), and scores the site by the mean of its leave-one-out
corrected marginal percentile ranks.
"""

import logging

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

from ._defaults import KDE_BW_FACTOR, MIN_NEIGHBOURHOOD_SIZE, N_ESTIMATORS, RF_MIN_SAMPLES_LEAF, RF_N_JOBS
from ._proximity import compute_proximity_sparse, get_neighbourhood, train_rf

logger = logging.getLogger(__name__)


def _extract_pixels(ds, predictor_vars, outcome_vars, mask_var, mask_condition):
    """Extract pixel data from xr.Dataset as numpy arrays.

    Parameters
    ----------
    ds : xr.Dataset
    predictor_vars : list of str
    outcome_vars : list of str
    mask_var : str
        Name of the mask variable.
    mask_condition : callable
        Function applied to the mask array to produce a boolean mask.

    Returns
    -------
    X : ndarray of shape (n_pixels, n_predictors)
    Y : ndarray of shape (n_pixels, n_outcomes)
    lats : ndarray of shape (n_pixels,)
    lons : ndarray of shape (n_pixels,)
    mask_2d : ndarray of shape (n_lat, n_lon), boolean
    """
    mask_data = ds[mask_var].values
    if hasattr(mask_data, "compute"):
        mask_data = mask_data.compute()
    mask_2d = mask_condition(mask_data)

    lat_grid, lon_grid = np.meshgrid(
        ds.coords["lat"].values, ds.coords["lon"].values, indexing="ij"
    )
    lats = lat_grid[mask_2d]
    lons = lon_grid[mask_2d]

    X_cols = []
    for v in predictor_vars:
        vals = ds[v].values
        if hasattr(vals, "compute"):
            vals = vals.compute()
        X_cols.append(vals[mask_2d])
    X = np.column_stack(X_cols)

    Y_cols = []
    for v in outcome_vars:
        vals = ds[v].values
        if hasattr(vals, "compute"):
            vals = vals.compute()
        Y_cols.append(vals[mask_2d])
    Y = np.column_stack(Y_cols)

    return X, Y, lats, lons, mask_2d


def _score_single_site(y_test, Y_ref_neighbourhood, weights, kde_bw_factor):
    """Compute condition score for a single test site via marginal KDE scoring.

    Fits d independent 1D weighted KDEs (one per outcome dimension), computes
    LOO-corrected percentile ranks, and combines via arithmetic mean.

    Parameters
    ----------
    y_test : ndarray of shape (n_outcomes,)
    Y_ref_neighbourhood : ndarray of shape (n_neighbours, n_outcomes)
    weights : ndarray of shape (n_neighbours,)
    kde_bw_factor : float

    Returns
    -------
    condition : float in [0, 1]
    """
    n_neighbours, d = Y_ref_neighbourhood.shape
    marginal_scores = np.empty(d)

    def _silverman_scaled(kde_obj):
        return kde_obj.silverman_factor() * kde_bw_factor

    for j in range(d):
        ref_vals = Y_ref_neighbourhood[:, j]
        test_val = y_test[j]

        kde_1d = gaussian_kde(ref_vals, bw_method=_silverman_scaled, weights=weights)

        d_test = kde_1d(np.array([test_val]))[0]
        d_refs = kde_1d(ref_vals)

        # LOO correction: remove self-kernel from each reference density
        h = kde_1d.factor * np.sqrt(kde_1d.covariance[0, 0])
        k_zero = 1.0 / (h * np.sqrt(2 * np.pi))
        d_refs_loo = (d_refs - weights * k_zero) / (1.0 - weights)
        d_refs_loo = np.maximum(d_refs_loo, 0.0)

        marginal_scores[j] = np.sum(weights[d_refs_loo <= d_test])

    joint_score = np.mean(marginal_scores)
    condition = float(np.clip(joint_score / 0.5, 0.0, 1.0))
    return condition


class RFProximityScorer:
    """Random Forest proximity-weighted marginal KDE condition scorer.

    Trains a Random Forest on reference site covariates and outcomes,
    computes leaf-based proximity weights between test and reference sites,
    then scores each test site using weighted 1D marginal KDEs with
    leave-one-out correction.

    Parameters
    ----------
    n_estimators : int, default=1000
        Number of trees in the Random Forest.
    min_samples_leaf : int, default=5
        Minimum samples per leaf node.
    n_jobs : int, default=-1
        Parallel jobs for RF training (-1 = all cores).
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_neighbourhood_size : int, default=50
        Warning threshold for sparse neighbourhoods.
    kde_bw_factor : float, default=1.2
        Multiplier on Silverman bandwidth (>1 = smoother).
    """

    def __init__(self, n_estimators=N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                 n_jobs=RF_N_JOBS, random_state=None,
                 min_neighbourhood_size=MIN_NEIGHBOURHOOD_SIZE,
                 kde_bw_factor=KDE_BW_FACTOR):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.min_neighbourhood_size = min_neighbourhood_size
        self.kde_bw_factor = kde_bw_factor

    def fit(self, ds, predictor_vars, outcome_vars):
        """Train RF on reference pixels (where ref_mask == 1).

        Parameters
        ----------
        ds : xr.Dataset
            Must contain variables listed in predictor_vars and outcome_vars,
            plus a ``ref_mask`` variable (1 = reference pixel).
        predictor_vars : list of str
            Names of predictor (covariate) variables.
        outcome_vars : list of str
            Names of outcome variables.

        Returns
        -------
        self
        """
        X_ref, Y_ref, lats, lons, _ = _extract_pixels(
            ds, predictor_vars, outcome_vars, "ref_mask",
            lambda m: m == 1,
        )
        self.X_ref_ = X_ref
        self.Y_ref_ = Y_ref
        self.predictor_vars_ = list(predictor_vars)
        self.outcome_vars_ = list(outcome_vars)

        self.rf_ = train_rf(
            X_ref, Y_ref,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return self

    def score(self, ds, predictor_vars, outcome_vars):
        """Score test pixels (where test_mask != 0).

        Parameters
        ----------
        ds : xr.Dataset
            Must contain variables listed in predictor_vars and outcome_vars,
            plus a ``test_mask`` variable (non-zero = test pixel, value = true
            condition).
        predictor_vars : list of str
        outcome_vars : list of str

        Returns
        -------
        result : xr.DataArray
            Condition scores with dims (lat, lon). NaN at non-test pixels.
        """
        X_test, Y_test, lats, lons, test_mask_2d = _extract_pixels(
            ds, predictor_vars, outcome_vars, "test_mask",
            lambda m: m != 0,
        )
        n_test = X_test.shape[0]

        logger.info("Computing proximity matrix (%d test x %d ref)...",
                     n_test, self.X_ref_.shape[0])
        prox = compute_proximity_sparse(self.rf_, self.X_ref_, X_test)

        logger.info("Scoring %d test sites...", n_test)
        conditions = np.empty(n_test)
        for i in range(n_test):
            ref_indices, weights = get_neighbourhood(
                prox, i, min_neighbourhood_size=self.min_neighbourhood_size,
            )
            Y_neighbourhood = self.Y_ref_[ref_indices]
            conditions[i] = _score_single_site(
                Y_test[i], Y_neighbourhood, weights, self.kde_bw_factor,
            )

        # Reconstruct 2D DataArray
        result = np.full(test_mask_2d.shape, np.nan)
        result[test_mask_2d] = conditions

        return xr.DataArray(
            result,
            dims=("lat", "lon"),
            coords={"lat": ds.coords["lat"], "lon": ds.coords["lon"]},
            name="condition",
        )
