"""HCAS condition scorer faithful to Williams et al. (2023) v2.1-3.

Implements the complete HCAS benchmarking workflow:
  1. Train RF to predict outcomes from covariates
  2. Build 2D reference-distance density surface P_ref(d_obs | d_pred)
  3. For each test site, select benchmarks via geographic + content filtering
  4. Score using Half-Cauchy weighted probabilities with LDC (Equations 2-3)
  5. Calibrate to [0, 1] via reference site scoring

All default parameters match Table S14 of the manuscript.
"""

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from ._defaults import (
    HCAS_BIN_SIZE,
    HCAS_CALIBRATION_SUBSAMPLE,
    HCAS_CAUCHY_LAMBDA,
    HCAS_DENSITY_SUBSAMPLE,
    HCAS_GEO_RADIUS_KM,
    HCAS_N_BENCHMARKS,
    HCAS_N_CANDIDATES,
    HCAS_N_TRUNCATE_BINS,
    HCAS_OMEGA,
    N_ESTIMATORS,
    RF_MIN_SAMPLES_LEAF,
    RF_N_JOBS,
)
from ._geo import haversine_km
from ._proximity import train_rf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pixels(ds, predictor_vars, outcome_vars, mask_var, mask_condition):
    """Extract pixel data from xr.Dataset as numpy arrays."""
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

    coords = np.column_stack([lats, lons])
    return X, Y, coords, mask_2d


def _standardize(Y, ref_mean=None, ref_std=None):
    """Zero-mean, unit-variance standardization."""
    if ref_mean is None:
        ref_mean = Y.mean(axis=0)
    if ref_std is None:
        ref_std = Y.std(axis=0)
        ref_std[ref_std == 0] = 1.0
    return (Y - ref_mean) / ref_std, ref_mean, ref_std


def _build_density_surface(Y_ref_std, Y_pred_ref_std, rng, *,
                           bin_size, n_truncate_bins, density_subsample):
    """Build the 2D normalised frequency surface P_ref(d_obs | d_pred).

    Faithful to manuscript:
    - Computes all unique reference-reference pairs from subsample
    - Uses fixed bin width Z (not fixed bin count)
    - Normalises columns, smooths with Moore neighbourhood (3x3 uniform filter)
    - Truncates to ~n_truncate_bins on each axis

    Returns (surface, d_obs_edges, d_pred_edges).
    """
    n_ref = Y_ref_std.shape[0]
    sub_n = min(density_subsample, n_ref)
    sub_idx = rng.choice(n_ref, size=sub_n, replace=False)
    Y_sub = Y_ref_std[sub_idx]
    Y_pred_sub = Y_pred_ref_std[sub_idx]

    # Compute ALL unique pairs (i < j) in chunks to manage memory
    logger.info("Computing all-pairs distances from %d subsampled reference sites...", sub_n)
    all_d_obs = []
    all_d_pred = []
    chunk_size = 500  # rows per chunk

    for start in range(0, sub_n, chunk_size):
        end = min(start + chunk_size, sub_n)
        for j_start in range(end, sub_n, chunk_size):
            j_end = min(j_start + chunk_size, sub_n)
            # Distances between rows [start:end] and rows [j_start:j_end]
            diff_obs = np.abs(Y_sub[start:end, np.newaxis, :] - Y_sub[np.newaxis, j_start:j_end, :])
            diff_pred = np.abs(Y_pred_sub[start:end, np.newaxis, :] - Y_pred_sub[np.newaxis, j_start:j_end, :])
            d_obs_chunk = diff_obs.sum(axis=2).ravel()
            d_pred_chunk = diff_pred.sum(axis=2).ravel()
            all_d_obs.append(d_obs_chunk)
            all_d_pred.append(d_pred_chunk)

        # Also within-chunk pairs where i < j
        if end - start > 1:
            block_obs = Y_sub[start:end]
            block_pred = Y_pred_sub[start:end]
            n_block = end - start
            for i_local in range(n_block):
                if i_local + 1 < n_block:
                    diff_o = np.abs(block_obs[i_local] - block_obs[i_local + 1:])
                    diff_p = np.abs(block_pred[i_local] - block_pred[i_local + 1:])
                    all_d_obs.append(diff_o.sum(axis=1))
                    all_d_pred.append(diff_p.sum(axis=1))

    all_d_obs = np.concatenate(all_d_obs)
    all_d_pred = np.concatenate(all_d_pred)
    logger.info("Computed %d pairwise distances", len(all_d_obs))

    # Fixed bin width Z (manuscript: Z = 0.005)
    max_obs = all_d_obs.max()
    max_pred = all_d_pred.max()
    d_obs_edges = np.arange(0, max_obs + bin_size, bin_size)
    d_pred_edges = np.arange(0, max_pred + bin_size, bin_size)

    # 2D frequency histogram
    surface, _, _ = np.histogram2d(
        all_d_obs, all_d_pred, bins=[d_obs_edges, d_pred_edges]
    )

    # Normalise each column (d_pred bin) to sum to 1
    col_sums = surface.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    surface /= col_sums

    # Smooth with bilinear interpolation (Moore neighbourhood = 3x3 uniform filter)
    surface = uniform_filter(surface, size=3, mode="nearest")

    # Re-normalise columns after smoothing
    col_sums = surface.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    surface /= col_sums

    # Truncate to ~n_truncate_bins on each axis
    n_obs_trunc = min(surface.shape[0], n_truncate_bins)
    n_pred_trunc = min(surface.shape[1], n_truncate_bins)
    surface = surface[:n_obs_trunc, :n_pred_trunc]
    d_obs_edges = d_obs_edges[:n_obs_trunc + 1]
    d_pred_edges = d_pred_edges[:n_pred_trunc + 1]

    logger.info("Density surface shape: %s, d_obs range: [0, %.3f], d_pred range: [0, %.3f]",
                surface.shape, d_obs_edges[-1], d_pred_edges[-1])

    return surface, d_obs_edges, d_pred_edges


def _lookup_probabilities_batch(d_obs_arr, d_pred_arr, surface, d_obs_edges, d_pred_edges):
    """Vectorized probability lookup for arrays of (d_obs, d_pred) pairs."""
    obs_bins = np.searchsorted(d_obs_edges, d_obs_arr, side="right") - 1
    pred_bins = np.searchsorted(d_pred_edges, d_pred_arr, side="right") - 1

    n_obs_bins, n_pred_bins = surface.shape
    valid = (
        (obs_bins >= 0) & (obs_bins < n_obs_bins)
        & (pred_bins >= 0) & (pred_bins < n_pred_bins)
    )

    probs = np.zeros(len(d_obs_arr))
    probs[valid] = surface[obs_bins[valid], pred_bins[valid]]
    return probs


def _select_benchmarks(test_y_std, test_y_pred_std, test_coord,
                       Y_ref_std, Y_pred_ref_std, coords_ref,
                       surface, d_obs_edges, d_pred_edges,
                       *, geo_radius_km, n_candidates, n_benchmarks):
    """Select HCAS benchmarks for a single test site.

    Manuscript algorithm:
    1. Geographic filter: R km radius
    2. Content filter: n_candidates closest in predicted distance
    3. Probability filter: n_benchmarks with highest p_ref

    Returns (bench_idx, bench_d_preds, bench_probs) or (None, None, None).
    """
    test_lat, test_lon = test_coord

    # Try progressively larger radii if needed
    for radius_mult in [1.0, 2.0, 4.0]:
        radius = geo_radius_km * radius_mult
        lat_tol = radius / 111.0 + 0.5
        lat_mask = np.abs(coords_ref[:, 0] - test_lat) <= lat_tol
        candidate_idx = np.where(lat_mask)[0]

        if len(candidate_idx) == 0:
            continue

        dists = haversine_km(
            test_lat, test_lon,
            coords_ref[candidate_idx, 0], coords_ref[candidate_idx, 1],
        )
        geo_mask = dists <= radius
        geo_idx = candidate_idx[geo_mask]

        if len(geo_idx) >= n_benchmarks:
            break
    else:
        geo_idx = np.arange(len(coords_ref))

    # Predicted Manhattan distances from test to candidates
    d_pred_candidates = np.sum(
        np.abs(test_y_pred_std - Y_pred_ref_std[geo_idx]), axis=1
    )

    # Select n_candidates with smallest d_pred
    n_cand = min(n_candidates, len(geo_idx))
    if n_cand >= len(geo_idx):
        top_cand_pos = np.arange(len(geo_idx))
    else:
        top_cand_pos = np.argpartition(d_pred_candidates, n_cand)[:n_cand]
    cand_idx = geo_idx[top_cand_pos]
    cand_d_pred = d_pred_candidates[top_cand_pos]

    # Observed Manhattan distances for candidates
    cand_d_obs = np.sum(
        np.abs(test_y_std - Y_ref_std[cand_idx]), axis=1
    )

    # Look up probabilities on density surface
    cand_probs = _lookup_probabilities_batch(
        cand_d_obs, cand_d_pred, surface, d_obs_edges, d_pred_edges
    )

    # Select n_benchmarks with highest probability
    n_bench = min(n_benchmarks, len(cand_idx))
    if n_bench >= len(cand_idx):
        top_prob_pos = np.arange(len(cand_idx))
    else:
        top_prob_pos = np.argpartition(cand_probs, -n_bench)[-n_bench:]
    bench_idx = cand_idx[top_prob_pos]
    bench_d_pred = cand_d_pred[top_prob_pos]
    bench_probs = cand_probs[top_prob_pos]

    return bench_idx, bench_d_pred, bench_probs


def _score_single_site_hcas(probs, d_preds, *, cauchy_lambda, omega):
    """Compute uncalibrated HCAS condition score (Equations 2-3).

    Equation 2:  w_i = 1 / (pi * (1 + (d_pred_i / lambda)^2))
    Equation 3:  H_c^LDC = omega * (sum(p_i * w_i) / sum(w_i) + p_max)
    """
    weights = 1.0 / (np.pi * (1.0 + (d_preds / cauchy_lambda) ** 2))
    w_sum = weights.sum()

    if w_sum == 0:
        return 0.0

    p_max = probs.max()
    score = omega * (np.sum(probs * weights) / w_sum + p_max)
    return float(score)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HCASScorer:
    """HCAS condition scorer (Williams et al. 2023, v2.1-3).

    Implements the HCAS benchmarking methodology with all parameters
    matching Table S14 of the manuscript by default.

    Parameters
    ----------
    n_estimators : int, default=1000
        Number of trees in the Random Forest.
    min_samples_leaf : int, default=5
        Minimum samples per leaf node.
    n_jobs : int, default=-1
        Parallel jobs for RF training.
    random_state : int or None, default=None
        Random seed for reproducibility.
    bin_size : float, default=0.005
        Z parameter: bin width for the 2D density surface.
    n_truncate_bins : int, default=400
        Truncate density surface to this many bins per axis after smoothing.
    n_candidates : int, default=50
        n_p parameter: initial candidate benchmarks (closest predicted distance).
    geo_radius_km : float, default=200.0
        R parameter: geographic search radius in km.
    n_benchmarks : int, default=20
        n_ref parameter: final benchmark count.
    cauchy_lambda : float, default=2.0
        Lambda parameter: half-Cauchy scale for distance weighting.
    omega : float, default=0.5
        Omega parameter: LDC confidence weight.
    density_subsample : int, default=100_000
        Number of reference sites subsampled for density surface construction.
    calibration_subsample : int, default=5_000
        Number of reference sites scored for calibration.
    """

    def __init__(self, n_estimators=N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                 n_jobs=RF_N_JOBS, random_state=None,
                 bin_size=HCAS_BIN_SIZE, n_truncate_bins=HCAS_N_TRUNCATE_BINS,
                 n_candidates=HCAS_N_CANDIDATES, geo_radius_km=HCAS_GEO_RADIUS_KM,
                 n_benchmarks=HCAS_N_BENCHMARKS, cauchy_lambda=HCAS_CAUCHY_LAMBDA,
                 omega=HCAS_OMEGA, density_subsample=HCAS_DENSITY_SUBSAMPLE,
                 calibration_subsample=HCAS_CALIBRATION_SUBSAMPLE):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.bin_size = bin_size
        self.n_truncate_bins = n_truncate_bins
        self.n_candidates = n_candidates
        self.geo_radius_km = geo_radius_km
        self.n_benchmarks = n_benchmarks
        self.cauchy_lambda = cauchy_lambda
        self.omega = omega
        self.density_subsample = density_subsample
        self.calibration_subsample = calibration_subsample

    def fit(self, ds, predictor_vars, outcome_vars):
        """Train RF on reference pixels, build density surface, and calibrate.

        Parameters
        ----------
        ds : xr.Dataset
            Must contain predictor and outcome variables plus ``ref_mask``
            (1 = reference pixel). Coordinates must include ``lat`` and ``lon``.
        predictor_vars : list of str
        outcome_vars : list of str

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)

        X_ref, Y_ref, coords_ref, _ = _extract_pixels(
            ds, predictor_vars, outcome_vars, "ref_mask",
            lambda m: m == 1,
        )
        self.X_ref_ = X_ref
        self.Y_ref_ = Y_ref
        self.coords_ref_ = coords_ref
        self.predictor_vars_ = list(predictor_vars)
        self.outcome_vars_ = list(outcome_vars)

        # 1. Train RF
        logger.info("Training RF with %d estimators on %d reference sites...",
                     self.n_estimators, X_ref.shape[0])
        self.rf_ = train_rf(
            X_ref, Y_ref,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        # 2. Predict outcomes for reference sites
        Y_pred_ref = self.rf_.predict(X_ref)

        # 3. Standardize
        Y_ref_std, ref_mean, ref_std = _standardize(Y_ref)
        Y_pred_ref_std, _, _ = _standardize(Y_pred_ref, ref_mean, ref_std)
        self.ref_mean_ = ref_mean
        self.ref_std_ = ref_std
        self.Y_ref_std_ = Y_ref_std
        self.Y_pred_ref_std_ = Y_pred_ref_std

        # 4. Build density surface (manuscript-faithful)
        logger.info("Building density surface...")
        self.surface_, self.d_obs_edges_, self.d_pred_edges_ = _build_density_surface(
            Y_ref_std, Y_pred_ref_std, rng,
            bin_size=self.bin_size,
            n_truncate_bins=self.n_truncate_bins,
            density_subsample=self.density_subsample,
        )

        # 5. Calibrate using reference site scores
        self._calibrate(rng)

        return self

    def _calibrate(self, rng):
        """Score a subsample of reference sites to establish calibration range."""
        cal_n = min(self.calibration_subsample, len(self.Y_ref_std_))
        cal_idx = rng.choice(len(self.Y_ref_std_), size=cal_n, replace=False)

        logger.info("Calibrating on %d reference sites...", cal_n)
        cal_raw = np.zeros(cal_n)

        for j in range(cal_n):
            idx = cal_idx[j]
            bench_idx, bench_d_preds, bench_probs = _select_benchmarks(
                self.Y_ref_std_[idx],
                self.Y_pred_ref_std_[idx],
                self.coords_ref_[idx],
                self.Y_ref_std_, self.Y_pred_ref_std_, self.coords_ref_,
                self.surface_, self.d_obs_edges_, self.d_pred_edges_,
                geo_radius_km=self.geo_radius_km,
                n_candidates=self.n_candidates,
                n_benchmarks=self.n_benchmarks,
            )
            if bench_idx is not None:
                cal_raw[j] = _score_single_site_hcas(
                    bench_probs, bench_d_preds,
                    cauchy_lambda=self.cauchy_lambda, omega=self.omega,
                )

        # Manuscript: linearly scaled by the maximum value to range between 0 and 1
        self.score_min_ = float(np.min(cal_raw))
        self.score_max_ = float(np.max(cal_raw))
        if self.score_max_ <= self.score_min_:
            self.score_max_ = self.score_min_ + 1e-6

        logger.info("Calibration range: [%.6f, %.6f]", self.score_min_, self.score_max_)

    def score(self, ds, predictor_vars, outcome_vars):
        """Score test pixels using HCAS benchmarking.

        Parameters
        ----------
        ds : xr.Dataset
            Must contain predictor and outcome variables plus ``test_mask``
            (non-zero = test pixel). Coordinates must include ``lat`` and ``lon``.
        predictor_vars : list of str
        outcome_vars : list of str

        Returns
        -------
        result : xr.DataArray
            Condition scores with dims (lat, lon). NaN at non-test pixels.
        """
        X_test, Y_test, coords_test, test_mask_2d = _extract_pixels(
            ds, predictor_vars, outcome_vars, "test_mask",
            lambda m: m != 0,
        )
        n_test = X_test.shape[0]

        # Predict and standardize test outcomes
        Y_pred_test = self.rf_.predict(X_test)
        Y_test_std, _, _ = _standardize(Y_test, self.ref_mean_, self.ref_std_)
        Y_pred_test_std, _, _ = _standardize(Y_pred_test, self.ref_mean_, self.ref_std_)

        # Score each test site
        logger.info("Scoring %d test sites...", n_test)
        raw_scores = np.zeros(n_test)

        for i in range(n_test):
            bench_idx, bench_d_preds, bench_probs = _select_benchmarks(
                Y_test_std[i], Y_pred_test_std[i], coords_test[i],
                self.Y_ref_std_, self.Y_pred_ref_std_, self.coords_ref_,
                self.surface_, self.d_obs_edges_, self.d_pred_edges_,
                geo_radius_km=self.geo_radius_km,
                n_candidates=self.n_candidates,
                n_benchmarks=self.n_benchmarks,
            )
            if bench_idx is not None:
                raw_scores[i] = _score_single_site_hcas(
                    bench_probs, bench_d_preds,
                    cauchy_lambda=self.cauchy_lambda, omega=self.omega,
                )

        # Apply calibration
        conditions = np.clip(
            (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_),
            0.0, 1.0,
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
