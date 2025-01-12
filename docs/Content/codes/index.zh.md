# 代码部分

本节将详细说明代码的结构，训练数据和测试模型的存放位置，以及如何开始训练和推理。

代码的总体结构如下：

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

我们将主要讨论以下部分：

1. **`data` 文件夹**：存放训练数据。
2. **`docs` 文件夹**：包含LHAI用户手册。
3. **`LHAI` 文件夹**：存放项目代码。
4. **`saves` 文件夹**：存放已训练/推理的模型及其结果。