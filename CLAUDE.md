# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This is a C++/ROOT LHAASO source-analysis framework. The documented runtime/build environment is:

```bash
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
```

The Makefile expects `root-config`, ROOT Minuit, zlib, SLALIB (`SLALIB_LIBDIR` and `SLALIB_INCDIR`), and yaml-cpp. Several include/library paths are hardcoded to `/home/lhaaso/hushicong/MyEnv/YAML_CPP/...`, so local builds may require updating `Makefile` and the yaml-cpp include lines in the executable sources.

## Common commands

Build the currently selected executable:

```bash
make
```

`Makefile` selects the executable through `MAINSRCS`. In the current checkout it builds `Src_Convo_Template.cc`; `Src_Main.cc`, `Src_TSMap.cc`, and `Src_GetSBP.cc` are present but commented out in `MAINSRCS`.

Clean build artifacts:

```bash
make clean
make distclean
```

The Makefile has a `make test` target:

```bash
make test
```

It runs `./$< exam.dat exam.root`, but `exam.dat` is not present in this checkout and the active executable may not accept those arguments. Treat it as stale unless the fixture is supplied.

Run the main source-fitting analysis after building `Src_Main`:

```bash
./Src_Main config/Tutorial/Example1_Crab/Fit.yaml config/Tutorial/Example1_Crab/ParInit.yaml
./Src_Main config/Testrun_Crab/Fit.yaml config/Testrun_Crab/ParInit_1src.yaml
./Src_Main config/Testrun_Crab/Fit.yaml config/Testrun_Crab/ParInit_2src.yaml
./Src_Main config/Testrun_Crab/Fit.yaml config/Testrun_Crab/ParInit_3src.yaml
```

Run one TS-map segment after building `Src_TSMap`:

```bash
./Src_TSMap config/Testrun_Crab/Fit.yaml 0 Results/Testrun_Crab/3src/TSmap/TSmap_0.root
```

Run a surface-brightness profile after building `Src_GetSBP`:

```bash
./Src_GetSBP config/Testrun_Crab/Fit.yaml 2.5 0.1 output.root
```

Run the template-convolution / NJU AI workflow after building `Src_Convo_Template`:

```bash
cd Tools/NJU_AI
python3 npy2root.py
../../Src_Convo_Template Fit.yaml ParInit.yaml 0 0
python3 root2npy.py
```

The documented NJU AI check datasets can also be run from their subdirectories:

```bash
cd Tools/NJU_AI/energy_dependent_psf
bash run_response.sh
cd ../roi_psf_background
bash run_response.sh
cd ../poisson_fluctuation
bash run_response.sh
```

Post-processing utilities:

```bash
python3 Tools/ROOT2FITs.py
python3 Tools/ROOT2TXT.py
Tools/SigMap_Eqm2Gal
python3 Tools/DrawSED_auto.py --DirRes ../Results/Example1_Crab --ParRes ../Results/Example1_Crab/ParRes.yaml --srcname Crab
python3 Tools/Read_analysis.py -f <input-file> -m <mode>
```

`Tools/ROOT2FITs.py` and `Tools/ROOT2TXT.py` are hardcoded around `test.root` / `hSig` unless edited.

No lint or formatting command is defined in this repository.

## Architecture overview

The repository is organized around separate C++ executables that share header-only analysis components under `src/`:

- `Src_Main.cc`: global source fitting, flux points, TS calculations, upper limits, residual/significance outputs, and optional TS-map job-script generation.
- `Src_TSMap.cc`: TS-map calculation for one ROI/segment.
- `Src_GetSBP.cc`: surface-brightness profile generation.
- `Src_Convo_Template.cc`: convolution of source/template maps, used by `Tools/NJU_AI` for NPY-to-ROOT-to-NPY workflows.

The analysis is YAML-driven:

- `Fit.yaml` is parsed by `src/Src_Config.h`. It resolves `WorkDir`, WCDA/KM2A data YAMLs, detector activation and bin ranges, ROI include/exclude settings, fitting modes, TS-map settings, and output paths.
- `ParInit.yaml` describes sources (`SRC`) and diffuse Galactic emission (`DGE`). The executables parse it into `Src_Src` and `Src_DGE` objects.
- `src/Src_MorModel.yaml` and `src/Src_SEDModel.yaml` define the supported morphology and spectral model tags/formulas loaded by `src/Src_Model.h`.

Detector-specific logic is split by plugin:

- WCDA data, response, and likelihood code lives in `src/Plugin_WCDA/`.
- KM2A data, response, and likelihood code lives in `src/Plugin_KM2A/`.

`src/Src_Template.h` owns the combined model template. It separates analytic point/Gaussian sources, numerically convolved sources, template sources, and DGE components, then provides the component counts and parameter bookkeeping used by the fitting modes.

`src/Src_FittingMode.h` defines the primary fitting modes: parameter fit, flux point, source TS, source TS per bin, and flux upper limit. Fitting uses ROOT `TMinuit`; the executable-level `FCN` functions call WCDA/KM2A plugin likelihood methods and combine their log-likelihoods.

Typical `Src_Main` flow is:

1. Read `Fit.yaml` via `Src_Config::Readin`.
2. Initialize model registries and templates from `src/Src_MorModel.yaml` and `src/Src_SEDModel.yaml`.
3. Open the configured ROOT sky maps and initialize the ROI.
4. Load WCDA/KM2A maps, response files, and simulation inputs for active detectors.
5. Parse `ParInit.yaml` into source and DGE components.
6. Initialize detector fitting objects, exposure/livetime distributions, and fitting modes.
7. Run enabled modes from `Fit:` and write `ParRes.yaml`, residual/significance maps, convolution excess ROOT outputs, SED products, and optional TS-map jobs.

## Data and generated outputs

Example configurations live under `config/Tutorial/` and `config/Testrun_Crab/`. Detector data configuration YAMLs live under `config/Data/WCDA/` and `config/Data/KM2A/`; these typically point to external ROOT data/response files, not files fully contained in this checkout.

`Results/` contains generated example outputs. Avoid treating those files as source inputs unless a config explicitly points to them.

`Update.sh` copies code, configs, tools, and results from a hardcoded upstream path into the current working tree. It can overwrite local files, so do not run it unless the user explicitly asks for that sync.
