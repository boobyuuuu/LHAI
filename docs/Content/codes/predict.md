# 从git clone开始推理

`predict.py`存放地址:

```
LHAI/LHAI/modeling/predict.py
```

在了解了LHAI源代码结构：`data`,`docs`,`LHAI`,`saves` 文件夹的基本作用之后，就可以开始推理了。

这一页将会从 `git clone` 开始，叙述我们该准备哪些东西，如何开始进行推理。

1. 第一步：git clone

    ```bash
    git clone --branch Code https://github.com/boobyuuuu/LHAI.git
    ```

    这会下载Code branch下的所有代码于LHAI文件夹内。打开文件夹：

    ```bash
    cd LHAI
    ```

2. 检查LHAI文件夹的项目structure，是否包含完整：

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

3. 配置 python 环境

    首先确定我们的机器的宏观环境能够用于机器学习任务，我一般喜欢选择服务器配置：

    ```
    PyTorch  2.1.0
    Python  3.10(ubuntu22.04)
    Cuda  12.1
    ```

    当然，更高的配置也没有问题，只是这个配置最经典和稳定；

    然后，需要创建python虚拟环境，安装python环境：

    ```bash
    pip install -m requirements.txt
    ```

    这样环境方面就没有问题了。

4. 准备数据，并且放到数据文件夹 `data` 对应的类别文件夹内，修改 `config.py` 使之能够指向我们的数据

    比如根据 `config.py` 中的默认配置，我的泊松类型数据 `poisson_src_bkg.pkl.npy` 路径就是：
    
    ```python
    "data" / "POISSON" / poisson_src_bkg.pkl.npy
    ```

    在predict中对评估用到的数据变量有两个：

    ```python
    PRE_DATA_PATH = PROJ_ROOT / "data" / "POISSON"
    # 参数：如果需要指定不同的数据集的地址名
    PRE_DATA_NAME = "poisson_src_bkg.pkl.npy"
    # 参数：数据集名称
    ```

5. 准备神经网络代码，放到神经网络文件夹内，修改 `config.py` 使之能够引用我们的神经网络代码

    比如根据 `config.py` 中的默认配置，我的CNN类型神经网络代码 `CNN_EXP_0_1.py` 路径就是：

    ```python
    "LHAI" / "models" / CNN_EXP_0_1.py
    ```

    在predict中对推理用到的模型变量有三个：

    ```python
    PRE_MODEL_PATH = PROJ_ROOT / "saves" / "MODEL"
    PRE_MODEL_NAME = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
    RRE_MODEL = "CNN"
    ```

6. 确保训练好的模型已经保存在 `saves/MODEL` 文件夹内。

7. 开始推理。

    方法1：设置 `config.py` 中PREDICT的参数，在 `modeling` 文件夹中直接运行。

    ```bash
    python predict.py
    ```

    方法2：高端一点的直接通过 `Typer` 接口设置命令行参数


    具体请看 Code Section 的 LHAI 手册页面。

<p align='right'>by Zihang Liu</p>

