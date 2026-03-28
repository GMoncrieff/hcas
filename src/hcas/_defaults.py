"""Default parameter values from Williams et al. (2023) Table S14.

These are the HCAS v2.1-3 benchmarking algorithm parameters as defined
in the manuscript's supplemental material.
"""

# Random Forest
N_ESTIMATORS = 1000
RF_MIN_SAMPLES_LEAF = 5
RF_N_JOBS = -1

# RF Proximity / KDE scoring
MIN_NEIGHBOURHOOD_SIZE = 50
KDE_BW_FACTOR = 1.2

# HCAS density surface (Table S14)
HCAS_BIN_SIZE = 0.005           # Z: bin width for 2D frequency histogram
HCAS_N_TRUNCATE_BINS = 400      # Truncate surface to ~400x400 after smoothing
HCAS_DENSITY_SUBSAMPLE = 100_000  # Reference sites subsampled for density surface

# HCAS benchmark selection (Table S14)
HCAS_GEO_RADIUS_KM = 200.0     # R: geographic search radius (km)
HCAS_N_CANDIDATES = 50          # n_p: initial candidates (smallest d_pred)
HCAS_N_BENCHMARKS = 20          # n_ref: final benchmarks (highest p_ref)

# HCAS scoring (Table S14, Equations 2-3)
HCAS_CAUCHY_LAMBDA = 2.0        # lambda: half-Cauchy scale parameter
HCAS_OMEGA = 0.5                # omega: LDC confidence parameter

# HCAS calibration
HCAS_CALIBRATION_SUBSAMPLE = 5_000
