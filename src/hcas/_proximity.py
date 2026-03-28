"""Random Forest training and sparse proximity computation.

Computes RF leaf-based proximity weights between test and reference sites
using a memory-efficient tree-by-tree approach with sparse matrix accumulation.
"""

import logging

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

from ._defaults import MIN_NEIGHBOURHOOD_SIZE, N_ESTIMATORS, RF_MIN_SAMPLES_LEAF, RF_N_JOBS

logger = logging.getLogger(__name__)


def train_rf(X_ref, Y_ref, *, n_estimators=N_ESTIMATORS,
             min_samples_leaf=RF_MIN_SAMPLES_LEAF, n_jobs=RF_N_JOBS,
             random_state=None):
    """Train a multioutput Random Forest on reference data.

    Parameters
    ----------
    X_ref : ndarray of shape (n_ref, n_covariates)
    Y_ref : ndarray of shape (n_ref, n_outcomes)
    n_estimators : int
    min_samples_leaf : int
    n_jobs : int
    random_state : int or None

    Returns
    -------
    rf : fitted RandomForestRegressor
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_ref, Y_ref)
    logger.info("RF OOB R² = %.4f", rf.oob_score_)
    return rf


def compute_proximity_sparse(rf, X_ref, X_test):
    """Compute test-to-reference proximity matrix via tree-by-tree iteration.

    For each tree, finds which reference sites share a leaf node with each
    test site.  Accumulates co-occurrence counts in a sparse matrix, then
    normalises by the number of trees.

    Parameters
    ----------
    rf : fitted RandomForestRegressor
    X_ref : ndarray of shape (n_ref, n_covariates)
    X_test : ndarray of shape (n_test, n_covariates)

    Returns
    -------
    prox : sparse CSR matrix of shape (n_test, n_ref), values in [0, 1]
    """
    n_ref = X_ref.shape[0]
    n_test = X_test.shape[0]
    n_trees = len(rf.estimators_)

    prox = sparse.csr_matrix((n_test, n_ref), dtype=np.float32)

    batch_size = 50
    for batch_start in range(0, n_trees, batch_size):
        batch_end = min(batch_start + batch_size, n_trees)
        all_rows = []
        all_cols = []

        for tree in rf.estimators_[batch_start:batch_end]:
            ref_leaves = tree.apply(X_ref)
            test_leaves = tree.apply(X_test)

            sort_idx = np.argsort(ref_leaves)
            sorted_leaves = ref_leaves[sort_idx]

            unique_test_leaves, inverse = np.unique(test_leaves, return_inverse=True)

            for k, leaf in enumerate(unique_test_leaves):
                left = np.searchsorted(sorted_leaves, leaf, side="left")
                right = np.searchsorted(sorted_leaves, leaf, side="right")
                if left == right:
                    continue
                matching_refs = sort_idx[left:right]
                test_indices = np.where(inverse == k)[0]
                rows = np.repeat(test_indices, len(matching_refs))
                cols = np.tile(matching_refs, len(test_indices))
                all_rows.append(rows)
                all_cols.append(cols)

        if all_rows:
            all_rows = np.concatenate(all_rows)
            all_cols = np.concatenate(all_cols)
            data = np.ones(len(all_rows), dtype=np.float32)
            batch_prox = sparse.csr_matrix(
                (data, (all_rows, all_cols)), shape=(n_test, n_ref)
            )
            prox = prox + batch_prox

        logger.debug("Proximity: processed %d/%d trees", batch_end, n_trees)

    prox = prox.multiply(1.0 / n_trees)
    return prox


def get_neighbourhood(prox, test_idx, *, min_neighbourhood_size=MIN_NEIGHBOURHOOD_SIZE):
    """Extract the neighbourhood for a single test site.

    Parameters
    ----------
    prox : sparse CSR matrix (n_test, n_ref)
    test_idx : int
    min_neighbourhood_size : int
        Warn if fewer neighbours than this.

    Returns
    -------
    ref_indices : ndarray of shape (n_neighbours,)
    weights : ndarray of shape (n_neighbours,), normalised to sum to 1
    """
    row = prox.getrow(test_idx)
    ref_indices = row.indices.copy()
    weights = np.asarray(row.data, dtype=np.float64).copy()

    if len(ref_indices) < min_neighbourhood_size:
        logger.warning(
            "Test site %d has only %d neighbours (< %d)",
            test_idx, len(ref_indices), min_neighbourhood_size,
        )

    w_sum = weights.sum()
    if w_sum > 0:
        weights /= w_sum

    return ref_indices, weights
