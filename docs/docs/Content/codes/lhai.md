# LHAI Project Code: LHAI Folder

The `LHAI` folder contains all the source code for this project, focusing on three main functionalities: training, predict, and testing.

## 1 Folder Structure

```plaintext
LHAI                        <- Source code for the project
│
├── __init__.py             <- Makes LHAI a Python module
│
├── config.py               <- Stores useful variables and configurations
│
├── features.py             <- Code for generating modeling features (currently unused)
│── plots.py                <- Code for generating visualizations (currently unused)
│
├── modeling                <- Code for model training and predict
│   ├── __init__.py
│   ├── predict.py          <- Code for model predict
│   └── train.py            <- Code for training models
│
├── models                  <- Contains neural network architectures
│   └── CNN_EXP_0_1.py      <- Naming convention: model_name_experiment_id.py
│
└── function                <- Utility functions used during training and predict
   ├── Dataset.py           <- Dataset utilities
   ├── Log.py               <- Logging utilities
   └── Loss.py              <- Loss functions
```

Next, we will discuss how to configure parameters, train models, and run predict.

## 2 Environment Variables: `config.py`

The `config.py` file contains all the environment variables used during training and predict.

### Training Parameters

```python
# ---- Train Parameters ----
EXP_NAME = "EXP_0_1"                                    # Parameter 1: Experiment identifier
MODEL_NAME = "CNN"                                      # Parameter 2: Model type (CNN, GAN, VAE, AE, DIFFUSION)
MODEL_PATH = PROJ_ROOT / "LHAI" / "models" / f"{MODEL_NAME}_{EXP_NAME}.py"
DATA_DIR = PROJ_ROOT / "data" / "POISSON"               # Parameter 3: Dataset directory
DATA_NAME = "poisson_src_bkg.pkl.npy"                   # Parameter 4: Dataset file name
DATA_PATH = DATA_DIR / DATA_NAME
SEED = 0                                                # Parameter 5: Random seed
TRAINTYPE = "poissonsrc+bkg_highresorig"                # Parameter 6: Training type
FRAC_TRAIN = 0.8                                        # Parameter 7: Train-test split ratio
EPOCHS = 400                                            # Parameter 8: Number of training epochs
BATCH_SIZE = 32                                         # Parameter 9: Batch size
LATENTDIM = 64                                          # Parameter 10: Latent dimension (VAE only)
LR_MAX = 5e-4                                           # Parameter 11: Maximum learning rate
LR_MIN = 5e-6                                           # Parameter 12: Minimum learning rate
```

### Predict Parameters

```python
# ---- Test Parameters ----
PRE_MODEL_PATH = PROJ_ROOT / "saves" / "MODEL"
PRE_DATA_PATH = PROJ_ROOT / "data" / "POISSON"
PRE_MODEL_NAME = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
PRE_MODEL = "CNN"
PRE_DATA_NAME = "poisson_src_bkg.pkl.npy"
PRE_SEED = 0
PRE_TRAINTYPE = "poissonsrc+bkg_highresorig"
PRE_FRAC_TRAIN = 0.8
PRE_BATCH_SIZE = 32
PRE_LATENT_DIM = 64
```

Before starting training or predict, ensure you understand each parameter and its purpose to avoid issues during execution.

!!! tip
    To modify parameters during execution, there are two approaches:
    
    1. Directly edit the values in `config.py`.
    2. Use command-line arguments when running `train.py` or `predict.py`, as explained below.

## 3 Training: `train.py`

This project uses the `Typer` library for managing command-line arguments. You can run the training script with the following command:

```bash
python train.py --model-name "GAN" --exp-name "MyExperiment" --data-dir "./data" --data-name "dataset.csv" --seed 42 --traintype "supervised" --frac-train 0.8 --epochs 10 --batch-size 32 --latentdim 128 --lr-max 0.01 --lr-min 0.001
```

Refer to **Section 2: Environment Variables** for details on parameter meanings.

!!! note
    1. Ensure the working directory in your terminal matches the script's location or use the full path to run the script.
    2. Default values from `config.py` will load into the `main` function if not provided in the command line.
    3. To check available parameters, use the following commands:
        ```bash
        python train.py --help
        python predict.py --help
        ```
    4. If only a specific parameter needs to be updated, you can pass it directly:
        ```bash
        python train.py --model-name "GAN"
        ```
        Other parameters will use their default values from `config.py`.

## 4 predict: `predict.py`

The process for running predict is identical to that of `train.py`. Use similar commands with relevant parameters for predict tasks.

<p align='right'>by Zihang Liu</p>