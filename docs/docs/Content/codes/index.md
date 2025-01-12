# Code Section

This section will explain the structure of our code, where to place training data, test models, and how to begin training and inference.

The overall structure of the code is as follows:

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data               <- Different types of data. (Recommended to upload in npy format)
│   ├── FERMI
│   ├── POISSON
│   ├── SIMU
│   └── RAW
│
├── saves              <- Figure/Loss saves, Trained and Serialized models, model predictions, etc.
│   ├── FIGURE         <- Figure saves during training.
│   │── PRE_FIG        <- Figure saves during prediction.
│   ├── LOSS           <- LOSS data (.npy) and LOSS figure (.png) saves during training.
│   └── MODEL          <- Trained and Serialized models.
│
├── docs               <- Default mkdocs project; see www.mkdocs.org for details
│
├── notebooks          <- Jupyter notebooks. Naming convention includes a number (for ordering),
│                         creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for LHAI and configuration
│                         for tools like black
│
├── references         <- Data dictionaries, manuals, and other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures for reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── LHAI               <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes LHAI a Python module
    │
    ├── config.py               <- Stores useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                <- Model training and inference code
    │   ├── __init__.py 
    │   ├── predict.py          <- Code for model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── models                  <- Neural network model code.
    │   └── CNN_EXP_0_1.py      <- Modelname_Experimentnumber.py
    │
    │── function                <- Functions used in training and prediction processes
    │  ├── Dataset.py           <- Function for loading datasets
    │  ├── Log.py               <- Function for logging
    │  └── Loss.py              <- Function for generating loss
    │
    └── plots.py                <- Code to create visualizations
```

We will mainly cover the following:

1. **The `data` folder**: Stores the training data.
2. **The `docs` folder**: Contains the LHAI user manual.
3. **The `LHAI` folder**: Contains the project code.
4. **The `saves` folder**: Contains the trained/inference models and results.