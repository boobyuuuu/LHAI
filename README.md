# LHAI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

NJU AI for LHAASO

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data               <- All project data.
│   ├── Evaluation     <- Evaluation datasets
│   ├── Original       <- Raw source data
│   └── Train          <- Training data
│
├── saves              <- Outputs such as models, predictions, and figures
│   ├── EVAL           <- Evaluation results
│   ├── MODEL          <- Trained model weights
│   ├── PREDICT        <- Prediction outputs
│   └── TRAIN          <- Training logs, losses, etc.
│
├── docs               <- MkDocs documentation source
│   ├── docs
│   │   ├── assets
│   │   │   ├── files
│   │   │   └── images
│   │   ├── blog
│   │   ├── changelog
│   │   ├── Content
│   │   │   ├── codes
│   │   │   ├── data
│   │   │   ├── eval
│   │   │   ├── network
│   │   │   └── others
│   │   ├── css
│   │   │   └── cards
│   │   ├── fonts
│   │   ├── js
│   │   ├── prompt
│   │   ├── resources
│   │   └── stylesheets
│   └── material
│       └── overrides
│           ├── assets
│           │   ├── images
│           │   └── stylesheets
│           └── zh
│               └── assets
│                   ├── images
│                   └── stylesheets
│
├── notebooks          <- Jupyter notebooks for exploration and documentation
├── references         <- Papers, books, and other reference materials
├── reports            <- Generated analysis as HTML, PDF, etc.
│   └── figures        <- Figures used in reports
│
├── requirements.txt   <- The requirements file for reproducing the environment
├── setup.cfg          <- Configuration file for flake8 and other tools
├── pyproject.toml     <- Project configuration (e.g., for black, isort)
│
└── codes              <- Source code for the LHAI project
    ├── config         <- Configuration definitions
    ├── function       <- Utility functions (e.g., Dataset.py, Log.py, Loss.py)
    ├── main           <- Training and evaluation entry points
    └── models         <- Model architectures
```

---

## Starting Model Training from Git Clone

After understanding the project structure, follow the steps below to start training and inference.

### 1. Git Clone

```bash
git clone --branch Code https://github.com/boobyuuuu/LHAI.git
cd LHAI
```

### 2. Verify Directory Structure

Ensure that the cloned directory structure matches the one shown above. Focus on:

* `codes/`: all source logic
* `data/`: all input data
* `saves/`: training outputs
* `docs/`: MkDocs-based documentation

### 3. Configure Python Environment

Recommended versions:

* Python: 3.10
* PyTorch: 2.1.0
* CUDA: 12.1

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Place training/evaluation data in `data/Train` or `data/Evaluation`. Example:

```python
"data" / "Train" / "xingwei_10000_64_train_v1.npy"
```

Update your config to point to the correct file path.

### 5. Start Training

Edit the configuration files in `codes/config` or run directly:

```bash
python codes/main/train.py
```

You can also invoke models dynamically (if Typer or argparse is supported):

```bash
python codes/main/train.py --model-name "CNN_EXP_0_1" --exp-name "exp1"
```

---

## Additional Notes

* Source code follows modular structure for clarity
* Git LFS is used to track large `.npy` files (e.g., training data)
* Documentation is written in MkDocs Material style, viewable at `/docs`
