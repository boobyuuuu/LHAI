# Experiment II: Close Two-Point-Source Separation — DDPM Inference and Resolution Evaluation Plan

## 1. Goal

Experiment II answers a single question:

> Given two point sources, how close in angular separation `D` and how unequal in flux ratio `Q = F2/F1`
> can they be before the DDPM reconstruction can no longer resolve them as two distinct sources?

The deliverable is a `Q x D` resolvability phase diagram on a fixed PSF/background/response, plus a
boundary curve (and tables) describing the resolution limit under the simplest scene assumption
(two isolated point sources, no other morphology).

Note on scene complexity: the answer in general depends on the sky-region complexity and source
composition. Experiment II fixes the simplest scene and only sweeps `(Q, D)`, so the resulting
boundary is the *best-case* resolution limit for this trained DDPM.

## 2. Inputs

### Environment

- Server/environment details: `env_info.txt` (Linux, Python 3.12, PyTorch 2.12 dev + CUDA 12.8, RTX 5090)
- Python dependencies: `requirements.txt`
- DDPM inference reference notebook: `codes/main/Exp1_ParametricMorphologyReconstruction/Exp1_DISK_DDPM_Evaluation.ipynb`

### Data

```text
data/Evaluation/Exp2_TwoPointSource/EXP2SEP_1000_64_poissonexcess.npy   shape (1000, 2, 64, 64)
data/Evaluation/Exp2_TwoPointSource/exp2_parameters.json
```

DDPM input format (standard two-channel paired dataset, identical convention to Exp1):

```text
data[:, 0]       = GT (point sources rasterized on a 64x64 grid, pixel size 0.1 deg)
data[:, 1]       = poissonexcess input (channel 5 of the 6-channel response)
```

Construction metadata in `exp2_parameters.json` (already verified):

- `global_settings.image_shape = [64, 64]`, `pixel_size_deg = 0.1`, `fov_deg = 6.4`
- `primary_flux_F1 = 1.0` (anchored at pixel `(NPIX//2, NPIX//2) = (32, 32)`)
- `Q_grid`: 200 log-uniform values on `[0.01, 1.0]`, endpoints included, descending (`Q[0]=1.0`, `Q[199]=0.01`)
- `D_grid_deg = [0.1, 0.2, 0.3, 0.4, 0.5]`, corresponding `separation_pixels = [1, 2, 3, 4, 5]`
- Secondary source sits `d_pix` pixels to the right of the primary on the horizontal mid-row
- Iteration order: outer loop over `D` (ascending), inner loop over `Q` (descending). Each `D`
  block occupies a contiguous 200-image range, e.g. `D=0.1 -> [0,199]`, `D=0.2 -> [200,399]`, ...,
  `D=0.5 -> [800,999]`
- Per-sample fields: `index`, `q_index`, `Q`, `F2` (with `F1 = 1.0` always)

Default model weight (same DDPM as Exp1):

```text
saves/MODEL/DDPM/Last_DDPM_KM2A_400epo_32bth_SR4poissonexcess.pth
```

Fixed KM2A PSF for `>25 TeV` (used as the resolution-criterion length scale):

```text
PSF r39 = 0.223607 deg
PSF r68 = 0.412311 deg
PSF r90 = 0.583095 deg
```

`DATA_RANGE = 2.0` (per-image min-max to `[-1, 1]`) is the same as the DDPM training/test setup
and must not be changed without retraining.

## 3. Notebook Layout

A single notebook covers the full sweep:

```text
codes/main/Exp2_TwoPointSource/Exp2_TwoPointSource_DDPM_Evaluation.ipynb
```

The notebook follows exactly the Exp1 six-step structure. Steps 1-3 are **copied verbatim** from
`codes/main/Exp1_ParametricMorphologyReconstruction/Exp1_DISK_DDPM_Evaluation.ipynb`; only the
constants identifying the dataset, run label, and metadata file are changed. Steps 4-6 are
Exp2-specific.

## 4. Notebook Workflow

### Step 1 — Load data and metadata (copy from Exp1_DISK_DDPM_Evaluation.ipynb)

Use the standard two-channel paired dataset where channel 0 is GT and channel 1 is the input map.

Reuse, without modification:

- `find_repo_root()` (walks from the notebook's directory up to the repo root containing both
  `codes/` and `data/`)
- `_ensure_nchw()`, `load_pair_arrays()` for `.npy` loading
- `show_available_files()` for sanity printing of available data and weights

Constants to change (and only these) relative to the DISK notebook:

```python
RUN_LABEL = "Exp2_TwoPointSource"

DATA_DIR = ADDR_ROOT / "data" / "Evaluation" / "Exp2_TwoPointSource"
DATA_NAME = "EXP2SEP_1000_64_poissonexcess.npy"

MODEL_WEIGHT_DIR = ADDR_ROOT / "saves" / "MODEL" / "DDPM"
MODEL_WEIGHT_NAME = "Last_DDPM_KM2A_400epo_32bth_SR4poissonexcess.pth"

EXP2_PARAM_JSON = DATA_DIR / "exp2_parameters.json"
```

`DIFFUSION_CONFIG_KEY`, `DIFFUSION_MODULE_NAME`, `DIFFUSION_CLASS_NAME`, `UNET_*` keys, and
`DATA_RANGE = 2.0` are kept identical to the DISK notebook. Expected loaded shapes:
`inputs_np.shape == (1000, 1, 64, 64)`, `targets_np.shape == (1000, 1, 64, 64)`.

### Step 2 — Run DDPM inference (copy from Exp1_DISK_DDPM_Evaluation.ipynb)

Reuse the DISK notebook's inference cells without modification:

- model class import (`import_model_class`), `ModelConfig` instantiation, `image_size` derivation
- `select_eval_indices(...)` with `EVAL_START=0`, `EVAL_STOP=None`, `EVAL_STEP=1`, `EVAL_INDICES=None`
  to evaluate all 1000 samples
- `run_evaluations(indices)` for batched `reverse_diffusion`
- `summarize_rows`, `print_normalized_summary`, `print_counts_summary`
- per-image min-max normalization with `DATA_RANGE` and the counts-space restoration step
- visualization helpers (`get_display_limits`, `visualize_indices`) and `plot_summary_bars`
  (kept for sanity plots; pick a few representative `VIS_INDICES` like the first/middle/last index
  of each `D` block, e.g. `[0, 100, 199, 200, 300, 399, 800, 900, 999]`)

This step must produce `generated_cache` (per-index reconstructed counts-space images),
`normalized_rows`, `counts_rows` exactly as the DISK notebook does.

### Step 3 — Extract GT and reconstructed peak parameters (adapt Exp1 style)

Two changes relative to the DISK notebook's "extract morphology" cells:

1. The metadata join uses `exp2_parameters.json` and a flat sample list rather than the
   per-source-type dict used in Exp1.
2. The image-measurement function is replaced by a two-peak detector instead of a single-source
   moment estimator.

#### 3.1 Build the per-sample metadata table

```python
exp2_meta = json.loads(EXP2_PARAM_JSON.read_text(encoding='utf-8'))
gs = exp2_meta['global_settings']
PIXEL_SIZE_DEG = float(gs['pixel_size_deg'])    # 0.1
NPIX = int(gs['image_shape'][0])                # 64
F1 = float(gs['primary_flux_F1'])               # 1.0

flat_samples = []  # length 1000, indexed by image_index
for sw in exp2_meta['sweeps']:
    D_deg = float(sw['D_deg'])
    sep_pix = int(sw['separation_pixels'])
    p_px = tuple(sw['primary_pixel_xy'])         # (32, 32)
    s_px = tuple(sw['secondary_pixel_xy'])       # (32 + sep_pix, 32)
    for s in sw['samples']:
        flat_samples.append({
            'image_index': int(s['index']),
            'D_deg': D_deg,
            'separation_pixels': sep_pix,
            'Q': float(s['Q']),
            'q_index': int(s['q_index']),
            'F1': F1,
            'F2': float(s['F2']),
            'x_GT1_pix': float(p_px[0]),
            'y_GT1_pix': float(p_px[1]),
            'x_GT2_pix': float(s_px[0]),
            'y_GT2_pix': float(s_px[1]),
        })
flat_samples.sort(key=lambda r: r['image_index'])
```

Convert GT pixel coordinates to angular coordinates in the same convention used in Exp1 image
measurement (pixel center = `0.5 + i`, axis in degrees `image_axis_deg = (np.arange(NPIX) + 0.5)
* PIXEL_SIZE_DEG`):

```python
def pix_to_deg(p):
    return (float(p) + 0.5) * PIXEL_SIZE_DEG
```

#### 3.2 Two-peak extraction function (used on GT and on SR with the same parameters)

```python
def extract_two_peaks(image_2d, x_GT1_pix, x_GT2_pix, y_GT_pix, sep_pix):
    """
    image_2d        : (H, W) counts-space image (DDPM SR or GT)
    x_GT1_pix       : primary GT x (pixel), here 32 (left peak)
    x_GT2_pix       : secondary GT x (pixel), here 32 + sep_pix (right peak)
    y_GT_pix        : shared mid-row y, here 32

    Returns dict with:
      x_peak1_pix, y_peak1_pix, P1   # primary peak
      x_peak2_pix, y_peak2_pix, P2   # secondary peak
      x_valley_pix, y_valley_pix, V  # global valley between the two peaks
      profile_y                      # 1-D profile along the mid-row used for valley search
      ok                             # False if peak extraction failed
    """
```

Detection rules (kept simple, deterministic, and the same on GT and SR so rasterization
systematics cancel):

- Search peak 1 in a small window around `x_GT1_pix`: `x in [x_GT1_pix - sep_pix/2, x_GT1_pix +
  sep_pix/2]`, `y in [y_GT_pix - 1, y_GT_pix + 1]`. Take the brightest pixel as `(x_peak1, y_peak1)`,
  with value `P1`.
- Search peak 2 in the symmetric window around `x_GT2_pix`. Take the brightest pixel as
  `(x_peak2, y_peak2)`, with value `P2`.
- Search valley `V` along the mid-row `y = y_GT_pix` between `x_peak1` and `x_peak2` (inclusive).
  If `x_peak1 == x_peak2` (same column), set `V = min(P1, P2)` and `D_valley = 0`.
- The function records all values in pixel and degree units. `peak_separation_pix = |x_peak2 - x_peak1|`.

The same function is invoked on:

- GT image `targets_np[i, 0]` -> `gt_peaks[i]` (validates rasterization and gives a per-sample
  reference for `D_valley_GT`)
- SR image `generated_cache[i]` (counts space) -> `sr_peaks[i]`

For the SR case, a sub-pixel peak refinement (parabolic fit on the 3-point neighbourhood along
each axis) is applied before reporting `x_peakK_pix`, `y_peakK_pix` to avoid quantization at
`D = 0.1 deg` (single-pixel separation).

Output: a `pandas.DataFrame` `peaks_df` indexed by `image_index` with the GT metadata columns,
plus the SR peak columns and the GT-peak columns.

### Step 4 — Define the separation criteria

This is Exp2-specific. Two criteria, applied independently per sample. A sample is called
"resolved" only if **both** are satisfied.

#### 4.1 Peak-position criterion

A reconstructed peak is "co-located" with a GT source if its angular distance to that GT source
is below half a PSF68:

```text
| x_peak1 - x_GT1 | < 0.5 * PSF68
| x_peak2 - x_GT2 | < 0.5 * PSF68
```

with the constants

```python
PSF39_DEG = 0.223607
PSF68_DEG = 0.412311
PSF90_DEG = 0.583095
PEAK_DIST_THRESHOLD_DEG = 0.5 * PSF68_DEG   # = 0.2061555 deg
```

In practice we use the 2-D Euclidean distance:

```text
center_error_1_deg = sqrt((x_peak1 - x_GT1)^2 + (y_peak1 - y_GT1)^2)
center_error_2_deg = sqrt((x_peak2 - x_GT2)^2 + (y_peak2 - y_GT2)^2)

peak_pos_correct = (center_error_1_deg < PEAK_DIST_THRESHOLD_DEG) AND
                   (center_error_2_deg < PEAK_DIST_THRESHOLD_DEG)
```

Notes:

- `0.5 * PSF68 ~= 0.206 deg ~= 2.06 pixels`. For `D = 0.1 deg` (1 pixel) the two GT sources are
  closer than the threshold, so this criterion alone is not sufficient — the valley criterion is
  what actually decides resolution at the smallest separations.
- We track each sub-criterion separately (`peak1_correct`, `peak2_correct`) so the failure mode
  (one peak vs both peaks) is recoverable from the saved table.

#### 4.2 Valley-depth criterion (Rayleigh-like)

Define

```text
D_valley = 1 - V / min(P1, P2)
```

where `P1`, `P2` are the two SR peak values and `V` is the minimum SR value on the mid-row
segment between the two peaks (inclusive). Bounds: `D_valley in [0, 1]`. `D_valley -> 0` means
the two peaks have merged into a flat ridge, `D_valley -> 1` means a clean dip between them.

Natural cuts (used in plots and tables):

```python
RESOLVE_LOOSE = 0.1   # D_valley > 0.1: marginally resolved
RESOLVE_MAIN  = 0.2   # D_valley > 0.2: resolved (primary criterion)
RESOLVE_TIGHT = 0.3   # D_valley > 0.3: clearly resolved
```

The primary resolution flag is

```text
valley_resolved = (D_valley > RESOLVE_MAIN)
```

The two looser/tighter cuts are reported alongside as sensitivity bands.

#### 4.3 Combined "resolved" flag

```text
resolved = peak_pos_correct AND valley_resolved
```

Per-sample columns added to `peaks_df`:

```text
center_error_1_deg, center_error_2_deg
peak1_correct, peak2_correct, peak_pos_correct
P1, P2, V, D_valley, D_valley_GT
valley_resolved, valley_resolved_loose, valley_resolved_tight
resolved
```

Sanity checks logged at the end of Step 4:

- For `D = 0.5 deg` and `Q = 1.0`, expect `resolved == True` for nearly every sample.
- For `D = 0.1 deg` and `Q = 0.01`, expect `resolved == False` for nearly every sample.
- `D_valley_GT` (computed on GT images) should be `~ 1.0` for `D >= 2 px` and may dip for `D = 1 px`
  due to single-pixel rasterization; this dip is the rasterization floor against which SR is
  compared.

### Step 5 — Apply criteria, aggregate, save outputs

Aggregation key is `(D_deg, Q)` (one cell per sample, since each `(D, Q)` pair appears exactly
once in the 1000-sample sweep). Bin the 200-point `Q` axis into a coarser grid for the phase
diagram while keeping the raw per-sample table for the boundary fit.

#### 5.1 Per-cell aggregation

```python
agg = peaks_df.groupby(['D_deg', 'Q']).agg(
    n_total=('image_index', 'count'),                # = 1 by construction
    resolved=('resolved', 'sum'),
    peak_pos_correct=('peak_pos_correct', 'sum'),
    valley_resolved=('valley_resolved', 'sum'),
    valley_resolved_loose=('valley_resolved_loose', 'sum'),
    valley_resolved_tight=('valley_resolved_tight', 'sum'),
    D_valley=('D_valley', 'mean'),
    D_valley_GT=('D_valley_GT', 'mean'),
    center_error_1_deg=('center_error_1_deg', 'mean'),
    center_error_2_deg=('center_error_2_deg', 'mean'),
).reset_index()
```

Because `n_total = 1` per cell, the "rate" is binary at this resolution. Two derived views are
built for plots:

- **Q-binned view**: bin `Q` into ~20 log-uniform bins (`np.logspace(-2, 0, 21)`), compute
  resolved fraction per `(D, Q_bin)`, plus binomial std error.
- **Q-smoothed view**: along each `D`, sort by `Q` (descending in original sweep order) and
  apply a rolling boolean fraction (window=11) of `resolved` for the boundary plot.

#### 5.2 Saved files

```text
saves/EVAL/Exp2_TwoPointSource/Exp2_per_sample.csv      # peaks_df, 1000 rows
saves/EVAL/Exp2_TwoPointSource/Exp2_per_cell.csv        # agg, 1000 rows
saves/EVAL/Exp2_TwoPointSource/Exp2_q_binned.csv        # 5 D x 20 Q bins
saves/EVAL/Exp2_TwoPointSource/Exp2_resolution_boundary.csv  # Q_threshold(D) for each cut
saves/EVAL/Exp2_TwoPointSource/Exp2_summary.json        # global numbers reported in Step 6
```

`Exp2_resolution_boundary.csv` columns (one row per `D` per cut):

```text
D_deg, criterion ('valley_0.1' | 'valley_0.2' | 'valley_0.3' | 'resolved_combined'),
Q_threshold, Q_threshold_lower, Q_threshold_upper, n_resolved, n_total
```

`Q_threshold(D)` is defined as the smallest `Q` (largest flux contrast at fixed `D`) for which
the rolling-fraction `resolved` rate stays above 0.5 down to that `Q`. A binomial 68% CI on the
fraction yields `Q_threshold_lower / Q_threshold_upper`.

### Step 6 — Answer the question (figures, tables, written summary)

Produce a fixed set of artifacts that together answer the experiment's question.

#### 6.1 Figures

Save under `saves/EVAL/Exp2_TwoPointSource/figures/`:

1. `fig01_QD_phase_resolved.png` — 2D heatmap, x = `Q` (log scale, 0.01..1.0), y = `D_deg`
   (0.1..0.5), color = `resolved` (binary, jittered) with binomial-bin overlay. PSF68 marked as
   horizontal reference.
2. `fig02_QD_phase_Dvalley.png` — same axes, color = mean `D_valley` with the three resolution
   contours (0.1, 0.2, 0.3) overlaid.
3. `fig03_resolution_boundary.png` — `Q_threshold(D)` curve for each of the four criterion
   variants, with 68% CI bands.
4. `fig04_profiles_grid.png` — 5x4 grid of mid-row 1-D profiles (rows = `D`, columns = four
   characteristic `Q` values: `1.0, 0.5, 0.1, 0.01`). Each panel overlays GT profile, SR profile,
   peak markers, and the valley point.
5. `fig05_visualize_examples.png` — six 2D images (GT vs SR side by side) at the chosen
   `(D, Q)` grid corners and on the boundary.
6. `fig06_center_error_vs_Q.png` — `center_error_1_deg` and `center_error_2_deg` vs `Q` per
   `D`, with the `0.5 * PSF68` threshold drawn as a horizontal line.

All plots use the same `D` color cycle and the `0.5 * PSF68 = 0.2061555 deg` threshold styling.

#### 6.2 Tables

- `table_resolution_boundary.csv`: for each `D in {0.1, 0.2, 0.3, 0.4, 0.5} deg`, the four
  `Q_threshold` values (`valley > 0.1/0.2/0.3` and combined `resolved`) with 68% CIs. This is the
  numerical answer to the question.
- `table_corner_cases.csv`: per-sample row dump for the corners
  `(D, Q) in {(0.1, 1.0), (0.1, 0.01), (0.5, 1.0), (0.5, 0.01)}` plus the row closest to each
  boundary, including all peak/valley fields.

#### 6.3 Written summary in `Exp2_summary.json` and the notebook's final markdown cell

The notebook's last cell prints and writes a concise answer:

- The smallest separation `D_min_eq` at which the DDPM resolves equal-flux pairs (`Q = 1.0`).
- For each `D`, the smallest `Q_min(D)` (largest flux contrast `1/Q_min`) at which the DDPM still
  resolves the pair.
- One or two sentences contextualizing the result against the PSF (e.g. comparing `D_min_eq` to
  `PSF68 = 0.412 deg`).
- An explicit caveat that this is the simplest two-source scene with no other morphology and
  fixed background; the boundary will shrink in more complex scenes.

## 5. Implementation Order

1. Copy Steps 1-3 cells from `Exp1_DISK_DDPM_Evaluation.ipynb` into
   `Exp2_TwoPointSource_DDPM_Evaluation.ipynb`. Change only the constants listed in Step 1 and
   replace the `exp1_meta` join with the `exp2_meta` flat-sample join described in 3.1.
2. Replace the DISK image-measurement function with `extract_two_peaks(...)` from 3.2 and run it
   on both GT and SR.
3. Implement Step 4 criteria as added DataFrame columns; verify the corner-case sanity checks.
4. Implement Step 5 aggregation and save the five output files.
5. Implement Step 6 figures, tables, and final markdown cell.
6. Smoke-run on a small `EVAL_STOP` (e.g. 50) before launching the full 1000-sample sweep.

## 6. Key Checks Before Final Reporting

- `inputs_np.shape == (1000, 1, 64, 64)` and `targets_np.shape == (1000, 1, 64, 64)` after Step 1.
- `DATA_RANGE = 2.0`, `DIFFUSION_CONFIG_KEY = "DDPM_64"`, `MODEL_WEIGHT_NAME` matches the file in
  `saves/MODEL/DDPM/`.
- Per-sample metadata join: every `image_index in [0, 999]` has exactly one matching entry in
  `flat_samples`, with the expected `D_deg` block (`[0,199] -> 0.1`, `[200,399] -> 0.2`, ...,
  `[800,999] -> 0.5`) and descending `Q`.
- `extract_two_peaks` is invoked with the same window and sub-pixel refinement on GT and SR.
- `D_valley` is bounded in `[0, 1]`; record any sample where `min(P1, P2) <= 0` and exclude it
  from valley-based aggregation (it goes into the "not resolved" bucket).
- For `D = 0.1 deg` GT images, expect `D_valley_GT < 1` (single-pixel rasterization floor); the
  SR threshold should be calibrated against this floor and not against `D_valley_GT = 1`.
- `PEAK_DIST_THRESHOLD_DEG = 0.5 * PSF68_DEG` is hard-coded once and reused everywhere.
- All saved CSV/JSON files include the `RUN_LABEL`, model weight name, threshold constants, and
  sample count in a header or sidecar JSON for traceability.
