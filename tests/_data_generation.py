"""Synthetic xarray Dataset generation for testing HCAS scorers.

Generates a 2D (lat, lon) Dataset with predictor/outcome variables,
ref_mask, and test_mask on a regular grid. Adapted from the original
data_generation.py with configurable sizes for fast tests.
"""

import numpy as np
import xarray as xr


# Spatial cluster centers (Australian regions)
SPATIAL_CLUSTER_CENTERS = [
    (-25.0, 135.0),  # Arid: central Australia
    (-37.0, 145.0),  # Temperate: SE Australia
    (-16.0, 146.0),  # Tropical: NE Australia
]
SPATIAL_SPREAD_KM = 400.0

CLUSTER_WEIGHTS = [0.40, 0.35, 0.25]
BIMODAL_FLIP_FRACTION = 0.30
TRUE_CONDITIONS = {"intact": 1.0, "mild": 0.7, "heavy": 0.2, "transformed": 0.0}
SHIFT_SDS = {"mild": 1.5, "heavy": 4.0, "transformed": 10.0}

N_COVARIATES = 5
N_OUTCOMES = 3


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _make_cluster_params():
    means = [
        np.array([2.0, -1.0, 0.5, 1.0, -0.5]),
        np.array([0.0, 0.5, 1.5, -0.5, 1.0]),
        np.array([-1.5, 1.5, 2.0, -1.0, 0.5]),
    ]
    covs = []
    for i in range(3):
        rng_cov = np.random.default_rng(i + 100)
        A = rng_cov.standard_normal((N_COVARIATES, N_COVARIATES)) * 0.3
        cov = A @ A.T + np.eye(N_COVARIATES) * 0.5
        cov *= 0.4
        covs.append(cov)
    return {"means": means, "covs": covs, "weights": np.array(CLUSTER_WEIGHTS)}


def _generate_covariates(n, cluster_params, rng):
    weights = cluster_params["weights"]
    cluster_labels = rng.choice(len(weights), size=n, p=weights)
    X = np.empty((n, N_COVARIATES))
    for k in range(len(weights)):
        mask = cluster_labels == k
        nk = mask.sum()
        if nk > 0:
            X[mask] = rng.multivariate_normal(
                cluster_params["means"][k], cluster_params["covs"][k], size=nk
            )
    return X, cluster_labels


def _compute_outcomes(X, cluster_labels, rng):
    n = X.shape[0]
    x0, x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    y0 = np.sin(x0 * x1) + 0.5 * x2 ** 2
    y1 = np.exp(-0.3 * x0) * x3 + np.tanh(x4)
    y2 = x0 * x1 * x2 / (1.0 + np.abs(x3))
    sigma = 0.3 + 0.5 * _sigmoid(x0)
    noise = rng.standard_normal((n, N_OUTCOMES)) * sigma[:, None]
    Y = np.column_stack([y0, y1, y2]) + noise
    alt_state_mask = np.zeros(n, dtype=bool)
    cluster0 = cluster_labels == 0
    n_cluster0 = cluster0.sum()
    if n_cluster0 > 0:
        flip = rng.random(n_cluster0) < BIMODAL_FLIP_FRACTION
        alt_indices = np.where(cluster0)[0][flip]
        alt_state_mask[alt_indices] = True
        Y[alt_indices, 0] += 4.0
    return Y, alt_state_mask


def _generate_spatial_coords(cluster_labels, rng):
    n = len(cluster_labels)
    coords = np.empty((n, 2))
    spread_deg = SPATIAL_SPREAD_KM / 111.0
    for k, (lat_c, lon_c) in enumerate(SPATIAL_CLUSTER_CENTERS):
        mask = cluster_labels == k
        nk = mask.sum()
        if nk > 0:
            coords[mask, 0] = rng.normal(lat_c, spread_deg, size=nk)
            coords[mask, 1] = rng.normal(lon_c, spread_deg, size=nk)
    return coords


def _apply_simple_degradation(Y_intact, categories, rng):
    """Apply degradation using global SD (simplified, no proximity needed)."""
    Y = Y_intact.copy()
    global_std = Y.std(axis=0)
    direction = np.array([-1.0, -1.0, 1.0])
    direction /= np.linalg.norm(direction)

    for i in range(len(Y)):
        cat = categories[i]
        if cat == "intact":
            continue
        magnitude = SHIFT_SDS[cat]
        shift = direction * magnitude * global_std
        noise = rng.standard_normal(N_OUTCOMES) * 0.15 * global_std
        Y[i] = Y_intact[i] + shift + noise
    return Y


def generate_test_dataset(n_ref=1000, n_test_per_category=100, seed=42):
    """Generate a synthetic xr.Dataset for testing.

    Parameters
    ----------
    n_ref : int
        Number of reference sites.
    n_test_per_category : int
        Number of test sites per degradation category (4 categories).
    seed : int
        Random seed.

    Returns
    -------
    ds : xr.Dataset
        Dataset with dims (lat, lon) containing predictor/outcome variables,
        ref_mask, and test_mask.
    """
    rng = np.random.default_rng(seed)
    cluster_params = _make_cluster_params()

    # Generate reference sites
    X_ref, cluster_ref = _generate_covariates(n_ref, cluster_params, rng)
    Y_ref, _ = _compute_outcomes(X_ref, cluster_ref, rng)
    coords_ref = _generate_spatial_coords(cluster_ref, rng)

    # Generate test sites
    categories_list = []
    true_cond_list = []
    for cat in ["intact", "mild", "heavy", "transformed"]:
        categories_list.extend([cat] * n_test_per_category)
        true_cond_list.extend([TRUE_CONDITIONS[cat]] * n_test_per_category)
    categories = np.array(categories_list)
    true_conditions = np.array(true_cond_list)
    n_test = len(categories)

    X_test, cluster_test = _generate_covariates(n_test, cluster_params, rng)
    Y_test_intact, _ = _compute_outcomes(X_test, cluster_test, rng)
    coords_test = _generate_spatial_coords(cluster_test, rng)

    # Apply simplified degradation (no proximity needed for test data gen)
    Y_test = _apply_simple_degradation(Y_test_intact, categories, rng)

    # Build a regular lat/lon grid that spans all sites
    all_lats = np.concatenate([coords_ref[:, 0], coords_test[:, 0]])
    all_lons = np.concatenate([coords_ref[:, 1], coords_test[:, 1]])

    n_total = n_ref + n_test
    # Grid size: aim for roughly sqrt(n_total * 1.5) per side
    grid_side = int(np.ceil(np.sqrt(n_total * 1.5)))

    lat_vals = np.linspace(all_lats.min() - 0.5, all_lats.max() + 0.5, grid_side)
    lon_vals = np.linspace(all_lons.min() - 0.5, all_lons.max() + 0.5, grid_side)

    n_lat = len(lat_vals)
    n_lon = len(lon_vals)

    # Assign sites to nearest grid cell
    def _assign_to_grid(site_lats, site_lons):
        lat_idx = np.searchsorted(lat_vals, site_lats) - 1
        lon_idx = np.searchsorted(lon_vals, site_lons) - 1
        lat_idx = np.clip(lat_idx, 0, n_lat - 1)
        lon_idx = np.clip(lon_idx, 0, n_lon - 1)
        return lat_idx, lon_idx

    # Initialize grid arrays
    predictor_grids = [np.full((n_lat, n_lon), np.nan) for _ in range(N_COVARIATES)]
    outcome_grids = [np.full((n_lat, n_lon), np.nan) for _ in range(N_OUTCOMES)]
    ref_mask_grid = np.zeros((n_lat, n_lon), dtype=np.float64)
    test_mask_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

    # Place reference sites
    ref_li, ref_lj = _assign_to_grid(coords_ref[:, 0], coords_ref[:, 1])
    for idx in range(n_ref):
        i, j = ref_li[idx], ref_lj[idx]
        for v in range(N_COVARIATES):
            predictor_grids[v][i, j] = X_ref[idx, v]
        for v in range(N_OUTCOMES):
            outcome_grids[v][i, j] = Y_ref[idx, v]
        ref_mask_grid[i, j] = 1.0

    # Place test sites (avoid overwriting ref sites by offsetting collisions)
    test_li, test_lj = _assign_to_grid(coords_test[:, 0], coords_test[:, 1])
    for idx in range(n_test):
        i, j = test_li[idx], test_lj[idx]
        # Skip if already a reference site
        if ref_mask_grid[i, j] == 1.0:
            # Find nearest empty cell
            for di in range(n_lat):
                found = False
                for dj in range(n_lon):
                    ni, nj = (i + di) % n_lat, (j + dj) % n_lon
                    if ref_mask_grid[ni, nj] == 0 and test_mask_grid[ni, nj] == 0:
                        i, j = ni, nj
                        found = True
                        break
                if found:
                    break
        for v in range(N_COVARIATES):
            predictor_grids[v][i, j] = X_test[idx, v]
        for v in range(N_OUTCOMES):
            outcome_grids[v][i, j] = Y_test[idx, v]
        test_mask_grid[i, j] = true_conditions[idx]

    # Handle test_mask == 0 for "transformed" category (true condition = 0.0)
    # Use a small epsilon so test_mask != 0 still identifies them
    test_mask_grid[(test_mask_grid == 0) & (ref_mask_grid == 0)] = 0.0
    # Re-place transformed sites with epsilon
    for idx in range(n_test):
        if categories[idx] == "transformed":
            i, j = test_li[idx], test_lj[idx]
            if ref_mask_grid[i, j] == 0:
                # Find the actual cell this was placed in
                pass
    # Simpler approach: use -1 flag or handle in test logic
    # Actually, use a very small positive value for transformed
    # Remap: place all test sites again tracking positions
    test_positions = []
    test_mask_grid[:] = 0.0
    used = set()
    for idx in range(n_test):
        i, j = int(test_li[idx]), int(test_lj[idx])
        # Find free cell
        while (i, j) in used or ref_mask_grid[i, j] == 1.0:
            j += 1
            if j >= n_lon:
                j = 0
                i = (i + 1) % n_lat
        used.add((i, j))
        test_positions.append((i, j))
        for v in range(N_COVARIATES):
            predictor_grids[v][i, j] = X_test[idx, v]
        for v in range(N_OUTCOMES):
            outcome_grids[v][i, j] = Y_test[idx, v]
        # Store true condition; use small epsilon for transformed (0.0 -> 1e-10)
        tc = true_conditions[idx]
        test_mask_grid[i, j] = tc if tc != 0.0 else 1e-10

    # Build Dataset
    data_vars = {}
    for v in range(N_COVARIATES):
        data_vars[f"covariate_{v}"] = (["lat", "lon"], predictor_grids[v])
    for v in range(N_OUTCOMES):
        data_vars[f"outcome_{v}"] = (["lat", "lon"], outcome_grids[v])
    data_vars["ref_mask"] = (["lat", "lon"], ref_mask_grid)
    data_vars["test_mask"] = (["lat", "lon"], test_mask_grid)

    ds = xr.Dataset(data_vars, coords={"lat": lat_vals, "lon": lon_vals})

    return ds
