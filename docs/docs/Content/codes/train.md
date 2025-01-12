# Starting Model Training from Git Clone

After understanding the basic functions of the LHAI source code structure (`data`, `docs`, `LHAI`, `saves` folders), you can begin training and inference.

This page explains how to prepare and start training, beginning with `git clone`.

---

### 1. Step One: Git Clone

Run the following command to clone the code from the `Code` branch:

```bash
git clone --branch Code https://github.com/boobyuuuu/LHAI.git
```

Navigate to the cloned repository:

```bash
cd LHAI
```

---

### 2. Verify the Project Structure

Ensure the `LHAI` folder contains the following structure:

```
├── LICENSE            <- Open-source license (if applicable)
├── Makefile           <- Utility commands, e.g., `make data` or `make train`
├── README.md          <- Top-level README for developers
├── data               <- Various types of data (recommended format: .npy)
│   ├── FERMI
│   ├── POISSON
│   ├── SIMU
│   └── RAW
├── saves              <- Stores images, loss data, trained models, and results
│   ├── FIGURE         <- Training images
│   ├── PRE_FIG        <- Inference images
│   ├── LOSS           <- Loss data and plots (.npy/.png)
│   └── MODEL          <- Saved trained models
├── docs               <- MkDocs project files (see www.mkdocs.org for details)
├── notebooks          <- Jupyter notebooks with standard naming conventions
├── pyproject.toml     <- Project configuration (e.g., Black formatting)
├── references         <- Data dictionaries, manuals, and other resources
├── reports            <- Generated reports (HTML, PDF, LaTeX)
│   └── figures        <- Figures and plots for reports
├── requirements.txt   <- Dependencies file (e.g., via `pip freeze`)
├── setup.cfg          <- flake8 configuration
└── LHAI               <- Source code
    ├── config.py               <- Configuration settings
    ├── dataset.py              <- Data generation/loading scripts
    ├── features.py             <- Feature generation
    ├── modeling                <- Model training and inference
    │   ├── predict.py          <- Inference code
    │   └── train.py            <- Training code
    ├── models                  <- Neural network models
    ├── function                <- Utilities for training/inference
    │   ├── Dataset.py          <- Dataset loader
    │   ├── Log.py              <- Logging utilities
    │   └── Loss.py             <- Loss functions
    └── plots.py                <- Visualization code
```

---

### 3. Configure Python Environment

Ensure your system meets the requirements for machine learning tasks. Recommended environment setup(not matter if close):

- **PyTorch**: 2.1.0
- **Python**: 3.10 (Ubuntu 22.04)
- **CUDA**: 12.1

Install dependencies using the provided file:

```bash
pip install -r requirements.txt
```

---

### 4. Prepare Data

Place your data in the appropriate folder under `data`. Update `config.py` to point to your dataset.

For example, if your Poisson data is named `poisson_src_bkg.pkl.npy`, its path in `config.py` should be:

```python
"data" / "POISSON" / "poisson_src_bkg.pkl.npy"
```

---

### 5. Prepare Neural Network Code

Place your neural network code in the `models` folder. Update `config.py` to reference your model.

For example, if your CNN model file is `CNN_EXP_0_1.py`, its path in `config.py` should be:

```python
"LHAI" / "models" / "CNN_EXP_0_1.py"
```

---

### 6. Pre-train for Debugging

Set a small number of `epochs` and a large `batch_size` to ensure the training process runs without issues.

---

### 7. Start Training

**Method 1**: Configure parameters in `config.py` and run:

```bash
python train.py
```

**Method 2**: Use the `Typer` command-line interface for dynamic parameter setting:

```bash
python train.py --model-name "GAN" --exp-name "MyExperiment" --data-dir "./data" --data-name "dataset.csv" --seed 42 --traintype "supervised" --frac-train 0.8 --epochs 10 --batch-size 32 --latentdim 128 --lr-max 0.01 --lr-min 0.001
```

For details, refer to the **Code Section** in the LHAI manual.

<p align='right'>by Zihang Liu</p>