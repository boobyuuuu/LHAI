# 代码部分

本节将详细说明代码的结构，训练数据和测试模型的存放位置，以及如何开始训练和推理。

代码的总体结构如下：

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data               <- Different types of data. (Recommended to upload with npy formation)
│   ├── FERMI
│   ├── POISSON
│   ├── SIMU
│   └── RAW
│
├── saves             <- Figure/Loss saves, Trained and Serialized models, model Predictions, .
│   ├── FIGURE        <- Figure saves in training.
│   │── PRE_FIG       <- Figure saves in predicting.
│   ├── LOSS          <- LOSS data(.npy) and LOSS Figure(.png) saves in training.
│   └── MODEL         <- Trained and Serialized models.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         LHAI and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`      
│
├── setup.cfg          <- Configuration file for flake8
│
└── LHAI   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes LHAI a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── evaluation.py              <- 3 different methods to evaluation trained models.
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── models                  <- Neural network model code.
    │   └── CNN_EXP_0_1.py      <- Modelname_Experimentnumber.py
    │
    └── function             <- Functions used in training and predicting processes.
       ├── Dataset.py        <- Function used in loading Dataset.
       ├── Log.py            <- Function used in Logging.
       └── Loss.py           <- Function used in Loss generating.

```

我们将主要讨论以下部分：

1. **`data` 文件夹**：存放训练数据。
2. **`docs` 文件夹**：包含LHAI用户手册。
3. **`LHAI` 文件夹**：存放项目代码。
4. **`saves` 文件夹**：存放已训练/推理的模型及其结果。