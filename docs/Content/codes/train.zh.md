# 从git clone开始训练模型

`train.py`存放地址:

```
LHAI/LHAI/modeling/train.py
```

在了解了LHAI源代码结构：`data`,`docs`,`LHAI`,`saves` 文件夹的基本作用之后，就可以开始训练模型、进行推理了。

这一页将会从 `git clone` 开始，叙述我们该准备哪些东西，如何开始进行训练。

1. 第一步：git clone

    先安装lfs，以下载大文件（数据）：

    ```
    conda install -c conda-forge git-lfs
    git lfs install
    ```

    ```bash
    git clone --branch Code https://github.com/boobyuuuu/LHAI.git
    ```

    这会下载Code branch下的所有代码于LHAI文件夹内。打开文件夹：

    ```bash
    cd LHAI
    ```

2. 检查LHAI文件夹的项目structure，是否包含完整：

    ```
    ├── LICENSE            <- 开源许可证（如果选择使用）
    ├── Makefile           <- 提供便利命令的Makefile，例如 `make data` 或 `make train`
    ├── README.md          <- 面向开发者的顶级README文件
    ├── data               <- 不同类型的数据（推荐使用npy格式上传）
    │   ├── FERMI
    │   ├── POISSON
    │   ├── SIMU
    │   └── RAW
    │
    ├── saves              <- 保存训练中的图像、损失数据、训练后的模型和预测结果
    │   ├── FIGURE         <- 训练过程中保存的图像
    │   │── PRE_FIG        <- 推理时保存的图像
    │   ├── LOSS           <- 训练过程中保存的损失数据(.npy)和损失图像(.png)
    │   └── MODEL          <- 保存的已训练模型和序列化模型
    │
    ├── docs               <- 默认的mkdocs项目文件；详细信息请参考 www.mkdocs.org
    │
    ├── notebooks          <- Jupyter notebooks，命名规范包括编号、创建者缩写和描述，例如：
    │                         `1.0-jqp-initial-data-exploration`
    │
    ├── pyproject.toml     <- 项目配置文件，包含LHAI的包元数据及工具（如black）的配置
    │
    ├── references         <- 数据字典、手册和其他说明性资料
    │
    ├── reports            <- 生成的分析报告（HTML、PDF、LaTeX等）
    │   └── figures        <- 报告中生成的图表和图形
    │
    ├── requirements.txt   <- 用于再现分析环境的依赖文件，例如通过
    │                         `pip freeze > requirements.txt` 生成
    │
    ├── setup.cfg          <- flake8的配置文件
    │
    └── LHAI               <- 本项目使用的源代码
        │
        ├── __init__.py             <- 使LHAI成为一个Python模块
        │
        ├── config.py               <- 存储有用的变量和配置
        │
        ├── dataset.py              <- 下载或生成数据的脚本
        │
        ├── features.py             <- 用于生成建模特征的代码
        │
        ├── modeling                <- 模型训练和推理代码
        │   ├── __init__.py 
        │   ├── predict.py          <- 用于模型推理的代码          
        │   └── train.py            <- 用于训练模型的代码
        │
        ├── models                  <- 神经网络模型代码
        │   └── CNN_EXP_0_1.py      <- 模型名称_实验编号.py
        │
        │── function                <- 训练和推理过程中使用的功能代码
        │  ├── Dataset.py           <- 数据集加载功能
        │  ├── Log.py               <- 日志记录功能
        │  └── Loss.py              <- 损失生成功能
        │
        └── plots.py                <- 用于生成可视化的代码
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

    在train中对训练用到的数据变量有三个：

    ```python
    DATA_DIR = PROJ_ROOT / "data" / "POISSON"                                           # 参数3：如果需要指定不同的数据集的文件名，可以在这里修改
    DATA_NAME = "poisson_src_bkg.pkl.npy"                                               # 参数4：如果需要指定不同的数据集，可以在这里修改
    DATA_PATH = DATA_DIR / DATA_NAME
    ```

5. 准备神经网络代码，放到神经网络文件夹内，修改 `config.py` 使之能够引用我们的神经网络代码

    比如根据 `config.py` 中的默认配置，我的CNN类型神经网络代码 `CNN_EXP_0_1.py` 路径就是：

    ```python
    "LHAI" / "models" / CNN_EXP_0_1.py
    ```

    在train中对训练用到的模型变量有三个：

    ```python
    EXP_NAME = "EXP_0_1"                                                                # 参数1：如果需要指定不同的实验，可以在这里修改
    MODEL_NAME = "CNN"                                                                  # 参数2：如果需要指定不同的模型，可以在这里修改
    MODEL_PATH = PROJ_ROOT / "LHAI" / "models" / f"{MODEL_NAME}_{EXP_NAME}.py"
    ```

6. 可以设置很小的 `eopch` 数以及 较大的 `batchsize`，进行快速的预训练，确保训练过程不会出问题。

7. 确保训练过程不会出问题之后，正式设置参数，开始训练。

    方法1：设置 `config.py` 中TRAIN的参数，在 `modeling` 文件夹中直接运行。

    ```bash
    python train.py
    ```

    方法2：高端一点的直接通过 `Typer` 接口设置命令行参数

    ```bash
    python train.py --model-name "GAN" --exp-name "MyExperiment" --data-dir "./data" --data-name "dataset.csv" --seed 42 --traintype "supervised" --frac-train 0.8 --epochs 10 --batch-size 32 --latentdim 128 --lr-max 0.01 --lr-min 0.001
    ```

    具体请看 Code Section 的 LHAI 手册页面。

<p align='right'>by Zihang Liu</p>