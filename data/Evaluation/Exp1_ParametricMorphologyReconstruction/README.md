# Exp1_ParametricMorphologyReconstruction

Experiment 1 builds single-image, single-source evaluation data for parametric morphology reconstruction.

## Fixed response assumptions

- Detector: KM2A
- Energy: `>25 TeV` (`emin=1.4`, `emax=3.4` in `log10(E/TeV)`)
- Response ROI DEC: `22 deg`
- Response ROI RA: sampled uniformly from `[0, 360)` by `generate_exp1_response.py`
- Spectrum: power law with `Epiv=50 TeV`, `alpha=3.0`
- Flux: response `F0 = np.geomspace(0.1, 10.0, 50)` with order `1e-16 cm^-2 s^-1 TeV^-1`, endpoints included

The GT `.npy` templates are unit-normalized. The LHAASO response program uses only `F0` in `ParInit.yaml` as the source flux, so the template sum is fixed to 1 and does not encode physical flux.

`Cx_deg` and `Cy_deg` are the source center coordinates inside the image. They are not the response ROI RA/DEC. Source centers are sampled from an independent Gaussian `N(0, 0.75 deg)` truncated to `|Cx| <= 1.6 deg` and `|Cy| <= 1.6 deg`, so most source flux remains inside the 6.4 deg field of view.

## Resolution convention

LHAASO response uses `0.1 deg / pixel`, so response inputs are `64 x 64`. For super-resolution analysis, this experiment first builds intrinsic templates on a `0.05 deg / pixel`, `128 x 128` grid, then merges each `2 x 2` block by summation into the `0.1 deg / pixel`, `64 x 64` grid. Both are saved.

## Confidence-map parameter grid

- `POINT`: no scale parameter, 50 flux values, total 50 images.
- `GAUSSIAN`: scale is `sigma_major_deg`.
- `DISK`: scale is `radius_deg`.
- `SHELL`: scale is `r_out_deg`; `r_in_deg` is randomized and recorded.
- `DIFFUSION`: scale is `r68_deg`; `r39_deg` is randomized and recorded.

For the four extended types, scale values are:

```text
[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6] deg
```

Each extended type has `10 scale values × 50 flux values = 500 images`. Other morphology parameters not used as confidence-map axes are randomized in reasonable ranges and saved in `exp1_parameters.json`.

## Local intrinsic GT generation

Run from the repository root:

```bash
python data/Evaluation/Exp1_ParametricMorphologyReconstruction/generate_exp1.py
```

Or from this directory:

```bash
python generate_exp1.py
```

This writes:

```text
EXP1POINT_50_64_GT.npy        (50, 64, 64)
EXP1POINT_50_128_GT.npy       (50, 128, 128)
EXP1GAUSSIAN_500_64_GT.npy    (500, 64, 64)
EXP1GAUSSIAN_500_128_GT.npy   (500, 128, 128)
EXP1DISK_500_64_GT.npy        (500, 64, 64)
EXP1DISK_500_128_GT.npy       (500, 128, 128)
EXP1SHELL_500_64_GT.npy       (500, 64, 64)
EXP1SHELL_500_128_GT.npy      (500, 128, 128)
EXP1DIFFUSION_500_64_GT.npy   (500, 64, 64)
EXP1DIFFUSION_500_128_GT.npy  (500, 128, 128)
```

It also writes:

```text
exp1_parameters.json
```

The JSON records the parameter-space construction strategy and every source's type, scale parameter, response F0, `Cx_deg`, `Cy_deg`, and array index.

## Server response generation

Place this folder at:

```text
/home/lhaaso/zhliu/LZH/response/Tools/NJU_AI/DATAGENERATING/Exp1_ParametricMorphologyReconstruction
```

The response driver expects the original response script to be available at:

```text
/home/lhaaso/zhliu/LZH/response/Tools/NJU_AI/DATAGENERATING/generate_response.py
```

On the server:

```bash
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
cd /home/lhaaso/zhliu/LZH/response/Tools/NJU_AI/DATAGENERATING/Exp1_ParametricMorphologyReconstruction
bash run_exp1_response.sh
```

To run only selected types:

```bash
bash run_exp1_response.sh --types POINT,GAUSSIAN
```

To prepare metadata and batch files without executing the response binary:

```bash
bash run_exp1_response.sh --dry-run --keep-work
```

Expected response outputs use the `64` GT arrays as input:

```text
EXP1POINT_50_64_RESPONSE.npy
EXP1GAUSSIAN_500_64_RESPONSE.npy
EXP1DISK_500_64_RESPONSE.npy
EXP1SHELL_500_64_RESPONSE.npy
EXP1DIFFUSION_500_64_RESPONSE.npy
```

Each response array has shape `(N, 7, 64, 64)` with channels:

```text
0 intrinsic/GT
1 excess
2 bkg
3 bkg_on
4 poisson
5 poissonbkg
6 poissonexcess
```

`generate_exp1_response.py` reads `exp1_parameters.json` directly, so response generation uses the exact `F0` grid from local GT generation instead of resampling flux values.
