"""Shared pytest fixtures for HCAS tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _data_generation import generate_test_dataset


@pytest.fixture(scope="session")
def synthetic_ds():
    """Small synthetic xr.Dataset for testing (session-scoped for speed)."""
    return generate_test_dataset(n_ref=800, n_test_per_category=80, seed=42)


@pytest.fixture(scope="session")
def var_names():
    """Predictor and outcome variable names."""
    return {
        "predictor_vars": [f"covariate_{i}" for i in range(5)],
        "outcome_vars": [f"outcome_{i}" for i in range(3)],
    }
