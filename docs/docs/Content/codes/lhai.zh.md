# LHAI项目代码：LHAI文件夹

`LHAI`文件夹存放了本项目的所有源代码，主要分为三个功能：训练、推理以及测试

## 1 Folder Structure

```
LHAI                        <- 本项目使用的源代码
│
├── __init__.py             <- 使LHAI成为一个Python模块
│
├── config.py               <- 存储有用的变量和配置
│
├── features.py             <- 用于生成建模特征的代码（目前没用）
│── plots.py                <- 用于生成可视化的代码（目前没用）
│
├── modeling                <- 模型训练和推理代码
│   ├── __init__.py 
│   ├── predict.py          <- 用于模型推理的代码          
│   └── train.py            <- 用于训练模型的代码
│
├── models                  <- 存储神经网络结构
│   └── CNN_EXP_0_1.py      <- 命名规则：模型名称_实验编号.py
│
└── function                <- 训练和推理过程中使用的函数，会在train.py以及predict.py用到
   ├── Dataset.py           <- 数据集
   ├── Log.py               <- 日志
   └── Loss.py              <- LOSS函数
```

接下来会叙述如何调用参数，进行训练，进行推理

## 2 环境变量：config.py

`config.py` 存储了训练、推理过程中用到的所有环境变量。

训练过程的所有参数：

```
# ---- Train Parameters ----
EXP_NAME = "EXP_0_1"                                    # 参数1：如果需要指定不同的实验，可以在这里修改
MODEL_NAME = "CNN"                                      # 参数2：模型类型名称（CNN,GAN,VAE,AE,DIFFUSION）
MODEL_PATH = PROJ_ROOT / "LHAI" / "models" / f"{MODEL_NAME}_{EXP_NAME}.py"
DATA_DIR = PROJ_ROOT / "data" / "POISSON"               # 参数3：数据集存储地址
DATA_NAME = "poisson_src_bkg.pkl.npy"                   # 参数4：数据集文件名称（数据集存储方法格式见Data Section）
DATA_PATH = DATA_DIR / DATA_NAME
SEED = 0                                                # 参数5：随机种子
TRAINTYPE = "poissonsrc+bkg_highresorig"                # 参数6：训练类型
FRAC_TRAIN = 0.8                                        # 参数7：训练集/测试集比例
EPOCHS = 400                                            # 参数8：训练轮数
BATCH_SIZE = 32                                         # 参数9：batch批次大小
LATENTDIM = 64                                          # 参数10：潜在维度(仅VAE模型)
LR_MAX = 5e-4                                           # 参数11：学习率上限
LR_MIN = 5e-6                                           # 参数12：学习率下限
```

推理过程的所有参数：

```
# ---- Test Parameters ----
PRE_MODEL_PATH = PROJ_ROOT / "saves" / "MODEL"
PRE_DATA_PATH = PROJ_ROOT / "data" / "POISSON"
PRE_MODEL_NAME = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
RRE_MODEL = "CNN"
PRE_DATA_NAME = "poisson_src_bkg.pkl.npy"
PRE_SEED = 0
PRE_TRAINTYPE = "poissonsrc+bkg_highresorig"
PRE_FRAC_TRAIN = 0.8
PRE_BATCH_SIZE = 32
PRE_LATENT_DIM = 64
```

在训练/推理开始前，一定要对每一个参数都非常了解，并且明确地知道自己训练/推理所需要的参数是什么，否则无法确保正确地完成了你的任务。

!!! tip

    在本项目中，如果想要修改程序运行中的环境变量/参数，一共有两种方法：

    1. 直接修改`config.py`中的参数

    2. python train.py --附加参数 ，这一个方法会在后面叙述

## 3 训练功能：train.py

本项目使用了 `Typer` 库为命令行接口设计的参数管理方式。以下是如何在 **命令行** 中运行你的代码并修改 `main` 参数的步骤：


```bash
python train.py --model-name "GAN" --exp-name "MyExperiment" --data-dir "./data" --data-name "dataset.csv" --seed 42 --traintype "supervised" --frac-train 0.8 --epochs 10 --batch-size 32 --latentdim 128 --lr-max 0.01 --lr-min 0.001
```

每个参数传递的具体含义，见 `2 环境变量：config.py` 的叙述。

!!! note

    1. 运行前确认当前终端的工作路径是脚本所在路径，或者使用绝对路径运行脚本。

        ```bash
        cd /path/to/train.py/folder
        python train.py --model-name "GAN" ...
        ```

    2. 如果你在 `LHAI.config` 中设置了默认值，例如 `EXP_NAME`, `MODEL_NAME`，这些值会作为默认值加载到 `main` 函数中。如果命令行中没有提供某些参数，将使用默认值。

    3. 如果不确定参数如何传递，可以运行以下命令查看帮助信息：

        ```bash
        python train.py --help
        python predict.py --help
        ```
        这个命令非常有用！


    4. 如果你只想修改模型名称，可以直接传递：

        ```bash
        python train.py --model-name "GAN"
        ```

        其余参数会使用 `config.py` 中的默认值。

## 4 推理功能：predict.py

具体运行方法与 `train.py` 完全一致。

<p align='right'>by Zihang Liu</p>