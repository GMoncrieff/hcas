# Demo Notebooks for HCAS Package

## Context

The `hcas` package implements two ecosystem condition assessment methods — `HCASScorer` and `RFProximityScorer` — but lacks end-to-end demo notebooks showing a real-world analytical workflow with Earth Engine data. These notebooks will serve as the primary usage documentation, demonstrating data acquisition from GEE, preprocessing via xee, model fitting, scoring, and result interpretation for a study area in South Africa's Western/Eastern Cape.

## Approach

Create **two self-contained Jupyter notebooks** in `notebook/`:
1. `hcas_demo.ipynb` — HCAS benchmarking method (Williams et al. 2023)
2. `rf_proximity_demo.ipynb` — RF proximity-weighted KDE method

Both share identical data acquisition/preprocessing but differ in the modelling section. Each notebook is fully self-contained.

## Study Area

- **Reference area** (`ref.geojson`): lon 17.77–23.59, lat -34.91–-32.35 (broader Western/Eastern Cape)
- **Test area** (`test.geojson`): lon 20.19–23.43, lat -34.49–-33.36 (subset nested within reference area)
- **Resolution**: 300m, EPSG:4326
- **Reference pixels**: HM < 0.05 within ref.geojson
- **Test pixels**: All within test.geojson; true condition = 1 - HM

## Notebook Structure (both follow the same layout)

### Cell Group 1: Setup & Authentication
- Imports: `ee`, `xee`, `xarray`, `numpy`, `matplotlib`, `json`, `hcas`
- Service account auth using `hm-30x30-9cf14c6efc4c.json` (same pattern as `xee_service.py`)
- `ee.Initialize(ee_creds)`

### Cell Group 2: Define Study Areas
- Load `ref.geojson` and `test.geojson` with `json.load()`
- Create `ee.Geometry.Rectangle()` from the bounding coordinates
- Create a combined geometry (union) for data loading extent
- Markdown cell explaining the reference/test area design

### Cell Group 3: Prepare GEE Datasets

**Predictors (6 variables):**

| Variable | GEE Source | Processing |
|----------|-----------|------------|
| `elevation` | `ee.Image("USGS/SRTMGL1_003")` | Select `elevation` band |
| `slope` | derived from SRTM | `ee.Terrain.slope(srtm)` |
| `aspect` | derived from SRTM | `ee.Terrain.aspect(srtm)` |
| `min_temp_coldest` | `ee.Image("WORLDCLIM/V1/BIO")` | Select `bio06` band (min temp coldest month, °C×10) |
| `annual_rainfall` | `ee.Image("WORLDCLIM/V1/BIO")` | Select `bio12` band (mean annual precipitation, mm) |
| `feb_rainfall` | `ee.Image("WORLDCLIM/V1/PREC")` | Select `02` band (February precipitation, mm) |

**Outcomes (3 variables):**

| Variable | GEE Source | Processing |
|----------|-----------|------------|
| `tree_homogeneity` | `ee.Image("projects/sat-io/open-datasets/PS_AFRICA_TREECOVER_2019_100m_V10")` | Compute focal stdDev in ~450m radius circle at native res, then transform: `1 / (1 + stdDev)`. Values 0–1 where 1 = perfectly uniform canopy cover. |
| `canopy_height` | `ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height")` | `.mosaic()` to create single image, select canopy height band |
| `lai_seasonality` | `ee.ImageCollection("NASA/VIIRS/002/VNP15A2H")` | Filter 2022, select `Lai` band, `.reduce(ee.Reducer.stdDev())` — magnitude of intra-annual LAI variability |

**True condition reference:**

| Variable | GEE Source | Processing |
|----------|-----------|------------|
| `human_modification` | `ee.ImageCollection("projects/sat-io/open-datasets/GHM/HM_2022_300M")` | `.mosaic()` — values 0 (pristine) to 1 (fully modified) |

Combine all into a single multi-band `ee.Image` using `.addBands()`.

### Cell Group 4: Convert to xarray via xee
- `xr.open_dataset(combined_image, engine='ee', scale=300, geometry=combined_geom, crs='EPSG:4326')`
- `.compute()` to materialize from lazy dask arrays
- Display dataset summary
- Optional: save to netCDF for caching (`ds.to_netcdf('hcas_data.nc')`)
- Markdown cell explaining xee bridge between GEE and xarray

### Cell Group 5: Create Masks & Clean Data
- **ref_mask**: pixels within ref.geojson bounds AND `human_modification < 0.05` → binary int
- **test_mask**: pixels within test.geojson bounds → value = `1 - human_modification` (true condition 0–1, where 1 = intact)
- Drop NaN rows: identify pixels where any predictor/outcome is NaN, set masks to 0 for those
- Add `ref_mask` and `test_mask` as variables to the dataset
- Print summary: number of reference pixels, test pixels, true condition distribution
- Quick map showing ref and test areas with their masks

### Cell Group 6: Fit Model
- **HCAS notebook**: `HCASScorer(n_estimators=500, random_state=42)` — use 500 trees for demo speed
- **RF-proximity notebook**: `RFProximityScorer(n_estimators=500, random_state=42)`
- Define `predictor_vars` and `outcome_vars` lists
- Call `scorer.fit(ds, predictor_vars, outcome_vars)`
- Print RF OOB R² score (`scorer.rf_.oob_score_`)
- Markdown cell explaining how the method works (different for each notebook)

### Cell Group 7: Score Test Area (uncalibrated)
- `result_raw = scorer.score(ds, predictor_vars, outcome_vars)`
- Print summary statistics: mean, median, std of predicted condition
- Markdown cell explaining the scoring process

### Cell Group 7b: Calibration (RF-proximity notebook only)
- Extract raw scores and true condition values from the test area:
  ```python
  test_mask = ds['test_mask'].values
  mask = test_mask != 0
  raw_scores = result_raw.values[mask]
  true_conds = test_mask[mask]
  ```
- Call `scorer.calibrate(raw_scores, true_conds)` — fits isotonic regression
- Re-score: `result = scorer.score(ds, predictor_vars, outcome_vars)`
- Print before/after MSE to show calibration improvement
- Markdown cell explaining:
  - Isotonic regression fits a monotonically non-decreasing step function
  - Ordering of scores is guaranteed to be preserved
  - Calibration shifts the score distribution to better match known true-condition labels
  - In practice, calibration labels would come from an independent validation set (e.g. field surveys), not the same test area — here we use the HM-derived values for demonstration

### Cell Group 8: Diagnostic — Expected vs Observed Outcomes
For 4 selected test cells (one per quartile of true condition):
- Use the fitted RF to predict "expected" outcomes: `scorer.rf_.predict(X_test_selected)`
- Compare with actual observed outcome values
- **Plot**: For each selected cell, a grouped bar chart showing expected (blue) vs observed (orange) for each outcome variable
- Annotate with the cell's true condition and predicted condition score
- Markdown explaining: deviation from expected = degradation signal

### Cell Group 9: Diagnostic — Score Distributions
- **Plot 1**: Histogram of predicted condition scores for all test pixels
- **Plot 2**: Scatter plot of true condition (1 - HM) vs predicted condition, with 1:1 line and Pearson r
- **Plot 3**: Box plots of predicted scores binned by true condition quartiles
- Compute and print correlation, RMSE, and MAE

### Cell Group 10: Maps — True vs Predicted Condition
- **Plot**: Side-by-side maps using `xr.DataArray.plot()`
  - Left: True condition (1 - HM) for test area, `cmap='RdYlGn'`, vmin=0, vmax=1
  - Right: Predicted condition for test area, same colormap and scale
- Crop to test area extent for display
- Markdown summarizing findings

## Key Implementation Details

### Extracting test data for diagnostics (Cell Group 8)
Both scorers store `X_ref_`, `Y_ref_`, `rf_` after fitting. To get test pixel data for diagnostics:
```python
# Extract test pixel arrays manually
test_mask_2d = ds['test_mask'].values != 0
X_test = np.column_stack([ds[v].values[test_mask_2d] for v in predictor_vars])
Y_test = np.column_stack([ds[v].values[test_mask_2d] for v in outcome_vars])
true_condition = ds['test_mask'].values[test_mask_2d]

# RF predictions for "expected under reference conditions"
Y_expected = scorer.rf_.predict(X_test)
```

### Mask creation from coordinate bounds
Since both geojsons are rectangular polygons:
```python
ref_coords = ref_geo['features'][0]['geometry']['coordinates'][0]
ref_lons = [c[0] for c in ref_coords]
ref_lats = [c[1] for c in ref_coords]

in_ref = (
    (ds.lon >= min(ref_lons)) & (ds.lon <= max(ref_lons)) &
    (ds.lat >= min(ref_lats)) & (ds.lat <= max(ref_lats))
)
```

### Files to create
- `notebook/hcas_demo.ipynb`
- `notebook/rf_proximity_demo.ipynb`

### Files to read (reuse patterns from)
- `notebook/xee_service.py` — auth pattern
- `notebook/ref.geojson`, `notebook/test.geojson` — study areas
- `src/hcas/_kde_scorer.py` — RFProximityScorer API (lines 117–234)
- `src/hcas/_hcas_scorer.py` — HCASScorer API (lines 286–501)
- `src/hcas/_proximity.py` — `compute_proximity_sparse`, `get_neighbourhood` (for advanced diagnostics)

## Verification

1. **Notebook structure**: Both notebooks should have clear markdown headers, explanatory text, and runnable code cells
2. **Data pipeline**: GEE → xee → xarray → hcas works end-to-end
3. **Outputs present**: Each notebook produces:
   - Expected vs observed outcome bar charts for 4 selected cells
   - Histogram of condition scores
   - True vs predicted scatter plot with correlation
   - Side-by-side maps of true and predicted condition
4. **API usage correct**: `fit(ds, predictor_vars, outcome_vars)` and `score(ds, predictor_vars, outcome_vars)` match the package API; dataset has `ref_mask` and `test_mask` variables with correct semantics
5. **Method-specific content**: HCAS notebook explains benchmarking/density surface; RF-proximity notebook explains leaf co-occurrence/KDE scoring + demonstrates the `calibrate()` step
6. **Climate data**: All climate variables sourced from WorldClim (no CHIRPS dependency)