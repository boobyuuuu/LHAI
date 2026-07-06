# Experiment I: Parametric Morphology Reconstruction — DDPM Inference and Confidence Evaluation Plan

## 1. Goal

Experiment I answers: under fixed PSF/background/response and given source type, flux, and scale, how reliable is the DDPM reconstruction in terms of detection rate, reconstruction accuracy, false positive rate, and morphology fidelity?

The final deliverable is a flux-r39 confidence phase diagram for each source type, plus one unified comparison figure containing all five types.

## 2. Inputs

### Environment

- Server/environment details: `env_info.txt`
- Python dependencies: `requirements.txt`
- DDPM model examples: `codes/main/evaluation_DDPM_TEST.ipynb`, `codes/main/predict_DDPM_CRAB.ipynb`

### Data

Experiment data are under:

```text
data/Evaluation/Exp1_ParametricMorphologyReconstruction/
```

DDPM input format (standard two-channel paired dataset):

```text
EXP1{TYPE}_{N}_64_poissonexcess.npy        shape (N, 2, 64, 64)
data[:, 0] = GT
data[:, 1] = poissonexcess (channel 5 from the original 6-channel response)
```

Construction metadata (flux, r39, Cx, Cy, ROI) come from:

```text
exp1_parameters.json
```

The original 6-channel response files `EXP1{TYPE}_{N}_64_RESPONSE.npy` are kept for reference but are **not** used by the DDPM evaluation notebook.

Default model weight:

```text
saves/MODEL/DDPM/Last_DDPM_KM2A_400epo_32bth_SR4poissonexcess.pth
```

Fixed KM2A PSF for `>25 TeV`:

```text
PSF r39 = 0.223607 deg
PSF r68 = 0.412311 deg
PSF r90 = 0.583095 deg
```

Other trained models/channels can be swapped manually, e.g. `excess`, `bkg_on`, `poisson_on`, or 128-size models.

## 3. Notebook Layout

Create five type-specific notebooks:

```text
Exp1_POINT_DDPM_Evaluation.ipynb
Exp1_GAUSSIAN_DDPM_Evaluation.ipynb
Exp1_DISK_DDPM_Evaluation.ipynb
Exp1_SHELL_DDPM_Evaluation.ipynb
Exp1_DIFFUSION_DDPM_Evaluation.ipynb
```

Create one summary notebook:

```text
Exp1_AllTypes_ConfidencePhaseDiagram.ipynb
```

Each type notebook should share the same structure and differ only in type-specific parameter extraction and correctness criteria.

## 4. Per-Type Notebook Workflow

### Step 1 — Load data and metadata

Use the standard two-channel paired dataset where channel 0 is GT and channel 1 is the input map. For Exp1, the inputs are the extracted `poissonexcess` channel from the original 6-channel response, saved as:

```text
EXP1{TYPE}_{N}_64_poissonexcess.npy
```

shape `(N, 2, 64, 64)`, with `data[:, 0] = GT` and `data[:, 1] = poissonexcess input`.

Also load the construction metadata for r39, Cx/Cy, ROI, etc. from:

```text
exp1_parameters.json
```

Expected response shape:

```text
POINT:      (50, 2, 64, 64)
EXTENDED:   (1000, 2, 64, 64)
```

Metadata provides response flux, ROI settings, `r39`, `Cx`, `Cy`, and construction parameters. For evaluation, any GT parameter that can be re-measured from the image should be measured from the GT image with the same function used on DDPM reconstructions; metadata values are retained as generation truth and response bookkeeping.

### Step 2 — Run DDPM inference

Strictly follow `codes/main/evaluation_DDPM_TEST.ipynb`. The data loading, per-image min-max normalization with `DATA_RANGE`, model loading, batched `reverse_diffusion`, and counts-space restoration are copied verbatim. Only the following are changed:

- `find_repo_root()` walks from the notebook's directory `codes/main/Exp1_ParametricMorphologyReconstruction/` up to the repo root that contains both `codes/` and `data/`.
- `DATA_DIR = ADDR_ROOT / 'data' / 'Evaluation' / 'Exp1_ParametricMorphologyReconstruction'`.
- `DATA_NAME` defaults to the corresponding `EXP1{TYPE}_..._poissonexcess.npy`.
- `MODEL_WEIGHT_NAME` defaults to `Last_DDPM_KM2A_400epo_32bth_SR4poissonexcess.pth`.
- `RUN_LABEL` is set per type, e.g. `Exp1_POINT`.
- After inference, the notebook also loads `exp1_parameters.json` to attach `flux_F0`, `r39`, `Cx`, `Cy`, ROI to per-sample metrics.

`DATA_RANGE = 2.0` is the same as the test notebook and must not be changed without retraining the model.


### Step 3 — Extract GT and reconstructed morphology parameters

Use the same post-processing and parameter extraction functions on GT maps and reconstructed maps whenever the parameter is image-measurable. This is required because the `64 x 64` grid has `0.1 deg / pixel` sampling, so drawing/rasterizing sources can introduce systematic offsets between construction metadata and the measurable image moments.

Use a consistent pipeline for all types:

1. Optional denoising / thresholding on reconstructed maps.
2. Detect source candidate region.
3. Estimate GT and reconstructed centers with the same center estimator.
4. Estimate type-specific GT and reconstructed morphology parameters with the same estimator.
5. Record failed detections explicitly.
6. For each reported metric, include statistical uncertainty and a grid/systematic error bar.

Type-specific parameter targets:

| Type | GT parameters | Reconstructed parameters |
| --- | --- | --- |
| POINT | `(Cx, Cy)` | `(Cx_hat, Cy_hat)` |
| GAUSSIAN | `(Cx, Cy, sigma_major, epsilon)` | `(Cx_hat, Cy_hat, sigma_major_hat, epsilon_hat)` |
| DISK | `(Cx, Cy, R_disk)` | `(Cx_hat, Cy_hat, R_disk_hat)` |
| SHELL | `(Cx, Cy, R_out, R_in)` | `(Cx_hat, Cy_hat, R_out_hat, R_in_hat)` |
| DIFFUSION | `(Cx, Cy, R39, R68)` | `(Cx_hat, Cy_hat, R39_hat, R68_hat)` |

For phase diagrams, use flux and `r39` as the two axes. For POINT, use flux only or place POINT as a one-row phase diagram.

### Step 4 — Compute correctness metrics

All morphology parameters should be min-max normalized to `[0, 1]` when comparing pure morphology errors.

Core metrics per sample:

```text
center_error_deg
center_error_over_psf68
parameter_error
normalized_parameter_error
grid_systematic_error_deg
total_error_with_grid_bar
detected: bool
correct_reconstruction: bool
false_positive_flag: bool
morphology_fidelity_score
```

Error bars:

- Statistical error comes from sample scatter within the same flux-r39 cell.
- Grid systematic error comes from comparing metadata construction parameters with values re-measured from the rasterized GT image.
- Report both separately when possible; otherwise use quadrature combination as the displayed error bar.

Recommended correctness thresholds:

- Center correct if `center_error < 0.3 * PSF68`, with `PSF68 = 0.412311 deg` for KM2A `>25 TeV`.
- Extension/radius correct if parameter error is below `0.3 * PSF68`, unless a type-specific stricter threshold is justified.
- GAUSSIAN eccentricity correct by normalized eccentricity error threshold.
- Detection rate = fraction of samples with a valid source candidate.
- Reconstruction confidence = fraction of samples in a flux-r39 bin satisfying all type-specific correctness criteria.

### Step 5 — Aggregate over flux-r39 grid

For each type, group samples by:

```text
flux_F0 × r39
```

Compute per-cell:

```text
detection_rate
correct_rate
false_positive_rate
median_center_error
median_parameter_error
morphology_fidelity_mean
n_samples
```

Save per-type metrics:

```text
saves/EVAL/Exp1_ParametricMorphologyReconstruction/{TYPE}/EXP1{TYPE}_metrics.csv
saves/EVAL/Exp1_ParametricMorphologyReconstruction/{TYPE}/EXP1{TYPE}_metrics.json
```

## 5. Confidence Definition

For each flux-r39 cell, define:

```text
confidence = correct_reconstruction_count / total_count
```

A region can be called reconstructable at 95% confidence if:

```text
confidence >= 0.95
```

The confidence boundary is the contour where confidence crosses 0.95 in the flux-r39 plane.

For sigma-language reporting, convert empirical confidence to Gaussian-equivalent sigma only as a reporting layer, not as the primary metric.

## 6. Unified Summary Notebook

`Exp1_AllTypes_ConfidencePhaseDiagram.ipynb` should:

1. Load all per-type metric files.
2. Plot one flux-r39 confidence map per extended type.
3. Plot POINT confidence versus flux.
4. Overlay 95% confidence contours.
5. Compare detection boundary and morphology fidelity across types.
6. Export final figures and tables.

Suggested outputs:

```text
saves/EVAL/Exp1_ParametricMorphologyReconstruction/summary/Exp1_alltypes_confidence_maps.png
saves/EVAL/Exp1_ParametricMorphologyReconstruction/summary/Exp1_alltypes_95pct_boundaries.csv
saves/EVAL/Exp1_ParametricMorphologyReconstruction/summary/Exp1_summary_metrics.json
```

## 7. Implementation Order

1. Build a reusable helper cell/module inside the first notebook for:
   - data loading
   - DDPM loading
   - batched inference
   - coordinate conversion
   - metric saving
2. Finish `POINT` notebook first because it only evaluates localization.
3. Extend the same workflow to `GAUSSIAN`.
4. Generalize parameter extraction for `DISK`, `SHELL`, and `DIFFUSION`.
5. Freeze metric file formats.
6. Build the unified summary notebook.

## 8. Key Checks Before Final Reporting

- Confirm response channel selection is correct: default channel `5 = poissonexcess`.
- Confirm DDPM input normalization matches training.
- Confirm reconstructed map coordinate system matches GT `(Cx, Cy)` convention.
- Confirm image-measurable GT parameters are re-extracted from GT images with the same functions used on reconstructions.
- Confirm grid/rasterization systematic uncertainties are included as error bars.
- Confirm PSF68 value used in thresholds is documented and fixed.
- Confirm failed detections are counted in denominators.
- Confirm each confidence-map cell has the expected number of samples.
