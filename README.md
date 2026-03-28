# hcas

Habitat Condition Assessment Scoring — two methods for estimating ecosystem condition from remotely sensed data and environmental covariates.

## Methods

**`RFProximityScorer`** — Random Forest proximity-weighted marginal KDE scoring. Defines ecological neighbourhoods via RF leaf co-occurrence, then scores each site using weighted 1D kernel density estimates with leave-one-out correction.

**`HCASScorer`** — The HCAS benchmarking method from Williams et al. (2023). Builds a 2D reference-distance density surface, selects benchmarks via geographic and content filtering, and scores using Half-Cauchy distance-weighted probabilities with Limited Degree of Confidence (LDC). All default parameters match Table S14 of the manuscript. See [hcas.md](hcas.md) for implementation details.

## Installation

```bash
pip install git+https://github.com/<owner>/hcas.git
```

For development:

```bash
git clone https://github.com/<owner>/hcas.git
cd hcas
pip install -e ".[test]"
```

## Quick start

Both scorers expect an `xarray.Dataset` with dimensions `(lat, lon)` containing:

- **Predictor variables** — environmental covariates (e.g. climate, topography)
- **Outcome variables** — remotely sensed ecosystem characteristics (e.g. vegetation indices)
- **`ref_mask`** — binary (0/1) indicating reference (intact) pixels used for model training
- **`test_mask`** — numeric, where non-zero values mark test pixels (the value can encode the true condition score for evaluation)

```python
from hcas import RFProximityScorer, HCASScorer

predictor_vars = ["temperature", "rainfall", "elevation"]
outcome_vars = ["ndvi", "fcover", "lai"]

# RF Proximity method
scorer = RFProximityScorer(n_estimators=1000, random_state=42)
scorer.fit(ds, predictor_vars, outcome_vars)
result = scorer.score(ds, predictor_vars, outcome_vars)
# result is an xr.DataArray with dims (lat, lon), NaN at non-test pixels

# HCAS method (manuscript defaults)
scorer = HCASScorer(n_estimators=1000, random_state=42)
scorer.fit(ds, predictor_vars, outcome_vars)
result = scorer.score(ds, predictor_vars, outcome_vars)
```

Both methods return an `xr.DataArray` with values in [0, 1], where 1 = reference condition and 0 = fully degraded. Non-test pixels are `NaN`.

## Dask support

The Dataset can be backed by dask arrays for scalable processing. During `fit()`, reference data is loaded into memory (sklearn requires numpy arrays). During `score()`, test pixel data is computed as needed.

```python
import xarray as xr

ds = xr.open_dataset("large_dataset.nc", chunks={"lat": 500, "lon": 500})
scorer.fit(ds, predictor_vars, outcome_vars)
result = scorer.score(ds, predictor_vars, outcome_vars)
```

## Parameters

### RFProximityScorer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 1000 | Number of RF trees |
| `min_samples_leaf` | 5 | Minimum samples per leaf |
| `n_jobs` | -1 | Parallel jobs for RF (-1 = all cores) |
| `random_state` | None | Random seed |
| `min_neighbourhood_size` | 50 | Warning threshold for sparse neighbourhoods |
| `kde_bw_factor` | 1.2 | Bandwidth multiplier (>1 = smoother KDE) |

### HCASScorer

| Parameter | Default | Manuscript symbol | Description |
|-----------|---------|-------------------|-------------|
| `n_estimators` | 1000 | — | Number of RF trees |
| `min_samples_leaf` | 5 | — | Minimum samples per leaf |
| `bin_size` | 0.005 | Z | Bin width for 2D density surface |
| `n_truncate_bins` | 400 | — | Surface truncation (→ ~400×400 matrix) |
| `n_candidates` | 50 | n_p | Candidate benchmarks (closest predicted distance) |
| `geo_radius_km` | 200.0 | R | Geographic search radius (km) |
| `n_benchmarks` | 20 | n_ref | Final benchmark count |
| `cauchy_lambda` | 2.0 | λ | Half-Cauchy scale parameter |
| `omega` | 0.5 | ω | LDC confidence weight |
| `density_subsample` | 100,000 | — | Reference sites subsampled for density surface |
| `calibration_subsample` | 5,000 | — | Reference sites scored for calibration |

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The `sandbox/` directory contains a script for running both methods on synthetic data:

```bash
cd sandbox/
python run_synthetic_test.py
```

## References

Williams KJ, Harwood TD, Ferrier S (2023). Habitat Condition Assessment System (HCAS) — methods v2.1-3.
