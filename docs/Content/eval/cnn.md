# CNN 类神经网络

## DEFAULT 参数

实验参数：

```
- 📦 实验名称                : EXP01
- 🧠 模型名称                : CNN
- 📁 模型脚本路径            : /root/LHAI/codes/models/CNN.py
- 📂 数据文件路径            : /root/LHAI/data/Train/xingwei_10000_64_train_v1.npy
- 📊 数据集切分比例          : 训练集 98.0% / 测试集 2.0%
- 📈 样本总数                : 10000
- 🔁 总训练轮数（Epochs）     : 400
- 📦 批次大小（Batch Size）  : 32
- 🌱 随机种子（Seed）        : 0
- 🔢 数据归一化范围          : 1.0
- 📉 学习率策略（Cosine）    : 最小 = 5.0e-06, 最大 = 5.0e-04
- 🧪 损失函数（Loss）        : msejsloss
- 🛠️ 优化器（Optimizer）     : AdamW
- 💻 使用设备（Device）      : cuda:0（NVIDIA GeForce RTX 4090）
- 📁 log保存地址             : /root/LHAI/saves/TRAIN/LOGS/trainlog_CNN
```

![LOSS分布图](Eval_loss_CNN_EXP01_jsdiv.png)

![Lineprofile图](Eval_distribution_CNN_EXP01.png)

![评估图](evaluation_plots_CNN_EXP01.png)

平均数据：

![alt text](image.png)

## dataprocess 参数

```
Average PSNR (SR): 18.2133
Average PSNR (Input): 13.2108
Average SSIM (SR): 0.2278
Average SSIM (Input): 0.2459
Average MS-SSIM (SR): 0.5334
Average MS-SSIM (Input): 0.5216
Average MAE (SR): 0.0427
Average MAE (Input): 0.1312
Average MSE (SR): 0.0157
Average MSE (Input): 0.0485
Average NRMSE (SR): 0.1241
Average NRMSE (Input): 0.2193
```

## DEFAULT Model - 400epochs

实验参数：

![alt text](image-1.png)

评估结果：

![alt text](image-2.png)

## CARN_v1

实验参数：

![alt text](image-3.png)

评估结果：

![alt text](image-4.png)

## CARN_v2

实验参数：

![alt text](image-5.png)

评估结果：

![alt text](image-6.png)

1. 看一下eval的input，每一次是否都相同，为什么会有0.1左右的波动

2. 对同一个模型的同一个参数，eval产生0.1的波动是正常现象。