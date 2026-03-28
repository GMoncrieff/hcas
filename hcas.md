# HCAS Implementation Details

This document describes how the `HCASScorer` class implements the HCAS benchmarking methodology from Williams et al. (2023), and how the default parameters map to the manuscript. It also explains where the implementation was corrected relative to earlier code to be faithful to the published method.

## Method overview

The HCAS method scores ecosystem condition by comparing each test site against a set of reference (intact) benchmarks. The workflow follows Box S2 of the manuscript:

1. **Train a Random Forest** on reference sites to predict ecosystem outcomes from environmental covariates
2. **Build a 2D density surface** P_ref(d_obs | d_pred) from all reference-reference site pairs, where d_obs and d_pred are Manhattan distances in observed and predicted outcome space
3. **For each test site**, select benchmarks via geographic filtering → content filtering → probability filtering
4. **Score** using Half-Cauchy distance-weighted probabilities with Limited Degree of Confidence (LDC)
5. **Calibrate** to [0, 1] by scoring a subsample of reference sites

## Parameter mapping to manuscript

All default parameter values match Table S14 of Williams et al. (2023):

| `HCASScorer` parameter | Default | Manuscript symbol | Manuscript description |
|-------------------------|---------|-------------------|------------------------|
| `bin_size` | 0.005 | Z | Bin width for frequency histogram axes |
| `n_candidates` | 50 | n_p | Number of candidate benchmarks (closest in predicted distance) |
| `geo_radius_km` | 200.0 | R | Geographic search radius in km |
| `n_benchmarks` | 20 | n_ref | Number of final benchmarks (highest probability) |
| `cauchy_lambda` | 2.0 | λ | Half-Cauchy scale parameter for distance decay weighting |
| `omega` | 0.5 | ω | LDC confidence weighting parameter |

Additional parameters not specified in the manuscript but needed for a practical implementation:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_truncate_bins` | 400 | Truncate density surface to ~400×400 after smoothing (manuscript states "truncated to approximate a 400×400 matrix") |
| `density_subsample` | 100,000 | Number of reference sites subsampled for density surface construction (computing all pairs from the full dataset may be prohibitive) |
| `calibration_subsample` | 5,000 | Number of reference sites scored to establish the calibration range |

## Equations

### Equation 2 — Half-Cauchy distance decay weights

For each benchmark *i* with predicted Manhattan distance d_pred_i from the test site:

```
w_i = 1 / (π × (1 + (d_pred_i / λ)²))
```

This is a half-Cauchy kernel that gives highest weight to benchmarks whose predicted characteristics are closest to the test site.

### Equation 3 — Limited Degree of Confidence (LDC) score

```
H_c^LDC = ω × (Σ(p_i × w_i) / Σ(w_i) + p_max)
```

Where p_i is the probability of each benchmark (from the density surface) and p_max is the maximum probability across all benchmarks. The ω parameter controls the overall confidence scaling.

## Density surface construction

The density surface P_ref(d_obs | d_pred) is built from reference-reference site pairs:

1. **Compute Manhattan distances** for all unique reference-reference pairs (i < j) in both observed and predicted outcome space (after standardisation)
2. **Bin into a 2D histogram** using fixed bin width Z = 0.005 on both axes (yielding a ~600×600 matrix for typical data)
3. **Normalise each column** (d_pred bin) to sum to 1, giving P(d_obs | d_pred)
4. **Smooth** using bilinear interpolation with a Moore neighbourhood — implemented as a 3×3 uniform filter
5. **Re-normalise columns** after smoothing
6. **Truncate** to approximately 400 bins on each axis to remove irrelevant large distances

## Benchmark selection

For each test site (Box S2, steps 6-8):

1. **Geographic filter**: Select reference sites within R km (default 200 km). If too few candidates, the radius expands progressively (×2, ×4, then all sites)
2. **Content filter**: From geographic candidates, select the n_p = 50 closest in predicted Manhattan distance
3. **Probability filter**: Compute each candidate's probability from the density surface P_ref(d_obs | d_pred), then select the n_ref = 20 with highest probability

## Calibration

The raw HCAS score is calibrated to [0, 1] by scoring a subsample of reference sites (which should score high, being intact). The manuscript states scores are "linearly scaled by the maximum value to range between 0 and 1":

```
condition = (raw_score - min_ref) / (max_ref - min_ref)
```

Clamped to [0, 1].

## Corrections from earlier code

The implementation corrects five deviations that existed in earlier prototype code to match the published manuscript:

### 1. Fixed bin width instead of fixed bin count

- **Earlier code**: Used `np.linspace(0, max, 201)` — 200 bins regardless of data range
- **Manuscript**: Z = 0.005 fixed bin width, producing ~600×600 bins for typical data
- **This implementation**: `np.arange(0, max_distance + bin_size, bin_size)`

### 2. Moore neighbourhood smoothing

- **Earlier code**: No smoothing applied after histogram construction
- **Manuscript** (line 826): "smoothed using bilinear interpolation (Moore neighbourhood at 0.005)"
- **This implementation**: `scipy.ndimage.uniform_filter(surface, size=3, mode="nearest")` followed by column re-normalisation. The Moore neighbourhood (3×3 grid of surrounding cells) is equivalent to a 3×3 uniform averaging filter

### 3. Post-construction truncation instead of pre-filtering

- **Earlier code**: Clipped distance range at the 99.5th percentile *before* building the histogram
- **Manuscript** (lines 827-828): Build the full surface first, *then* "truncated to remove irrelevant large distances to approximate a 400×400 matrix"
- **This implementation**: Builds the full histogram, smooths, then truncates to `n_truncate_bins=400` on each axis

### 4. All-pairs distances instead of random sampling

- **Earlier code**: Randomly sampled ~10M individual pairs with replacement
- **Manuscript** (lines 817-818): Two sets of Manhattan distances derived for "each reference-reference site-pair using the training data"
- **This implementation**: Computes all unique pairs (i < j) from a subsample, using chunked computation to manage memory. The `density_subsample` parameter controls the subsample size (default 100,000; reduce for faster computation)

### 5. Min/max calibration instead of percentile-based

- **Earlier code**: Used `score_max = median(cal_raw)`, `score_min = percentile(cal_raw, 1)`
- **Manuscript** (Table S2 v2.0): "linearly scaled by the maximum value to range between 0 and 1"
- **This implementation**: `score_min = min(cal_raw)`, `score_max = max(cal_raw)`

## Replicating the manuscript methodology

To use the exact parameters from the manuscript, simply use the defaults:

```python
from hcas import HCASScorer

scorer = HCASScorer(n_estimators=1000, random_state=42)
scorer.fit(ds, predictor_vars, outcome_vars)
result = scorer.score(ds, predictor_vars, outcome_vars)
```

All Table S14 parameters (Z=0.005, n_p=50, R=200 km, n_ref=20, λ=2.0, ω=0.5) are the defaults.

To adjust parameters for different study contexts:

```python
# Larger geographic search for sparse reference networks
scorer = HCASScorer(geo_radius_km=500.0)

# More benchmarks for noisier data
scorer = HCASScorer(n_benchmarks=50, n_candidates=100)

# Faster density surface construction (fewer pairs)
scorer = HCASScorer(density_subsample=10_000)

# Finer or coarser density surface
scorer = HCASScorer(bin_size=0.002)  # finer
scorer = HCASScorer(bin_size=0.01)   # coarser
```
