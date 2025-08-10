# 模型介绍与规范化标准

## 模型发展脉络

本篇介绍超分辨常用的神经网络技术，包括：

1. `CNN` 类网络原理与结构

    >> `VAE` 网络原理与结构

    >> `ResNet` 网络原理与结构

    >> `U-Net` 网络原理与结构

2. `GAN` 类网络原理与结构

4. `Transformer` 类网络原理与结构

5. `DIFFUSION` 类网络原理与结构

6. 特殊架构模型

```mermaid
graph TD
    %% ===== 基础架构 =====
    INTERP[1970s+ 插值方法<br>双线性/双三次] --> SPARSE[2000s 稀疏表示]
    SPARSE --> NEIGHBOR[2010 ANR<br>邻域嵌入回归]
    
    %% ===== CNN主线 =====
    SPARSE --> SRCNN[2014 SRCNN<br>首个端到端CNN]
    SRCNN --> VDSR[2016 VDSR<br>深度残差学习]
    VDSR --> DRCN[2016 DRCN<br>递归残差]
    VDSR --> EDSR[2017 EDSR<br>去BN的深残差]
    EDSR --> RCAN[2018 RCAN<br>通道注意力]
    RCAN --> SAN[2019 SAN<br>二阶注意力]
    
    %% ===== 轻量化分支 =====
    SRCNN --> FSRCNN[2016 FSRCNN<br>加速上采样]
    FSRCNN --> CARN[2018 CARN<br>级联残差]
    CARN --> IMDN[2019 IMDN<br>信息蒸馏]
    
    %% ===== GAN主线 =====
    SRCNN --> SRGAN[2017 SRGAN<br>感知质量突破]
    SRGAN --> ESRGAN[2018 ESRGAN<br>相对判别器]
    ESRGAN --> BSRGAN[2021 BSRGAN<br>盲超分]
    ESRGAN --> SwinGAN[2022 SwinGAN<br>Transformer判别器]
    
    %% ===== 注意力演进 =====
    RCAN --> HAN[2020 HAN<br>混合注意力]
    SAN --> NAF[2022 NAF<br>无激活网络]
    
    %% ===== Transformer主线 =====
    HAN --> TTSR[2020 TTSR<br>首篇超分Transformer]
    TTSR --> SwinIR[2021 SwinIR<br>窗口注意力]
    SwinIR --> EDT[2022 EDT<br>扩散增强]
    
    %% ===== Diffusion主线 =====
    DDPM[2020 DDPM<br>去噪扩散奠基] --> SR3[2021 SR3<br>首篇扩散超分]
    SR3 --> LDM[2022 LDM<br>潜在空间扩散]
    LDM --> DiffIR[2023 DiffIR<br>Transformer引导]
    
    %% ===== 其他独特架构 =====
    SPARSE --> KRR[2012 KRR<br>核岭回归]
    NEIGHBOR --> APlus[2014 A+<br>锚点邻域回归]
    APlus --> SelfEx[2015 SelfEx<br>自相似性]
    
    %% ===== 神经架构搜索 =====
    EDSR --> SRNAS[2019 SRNAS<br>神经架构搜索]
    SRNAS --> FALSR[2020 FALSR<br>轻量搜索]
    
    %% ===== 跨技术融合 =====
    SwinIR --> Restormer[2022 Restormer<br>多尺度Transformer]
    DiffIR --> DiffBIR[2024 DiffBIR<br>盲复原扩散]
    SwinGAN --> DiffGAN[2023 DiffGAN<br>扩散引导对抗]
    
    %% ===== 交叉连接 =====
    SelfEx --> EDSR
    ESRGAN --> RCAN
    LDM --> SwinIR
    NAF --> Restormer
    FALSR --> CARN
    KRR --> SRCNN
```

```mermaid
gantt
    title 超分辨率PSNR演进 (DIV2K ×4)
    dateFormat  YYYY
    axisFormat  %Y
    
    section 关键突破
    Bicubic : 2010, 28.42
    SC-SR : 2012, 29.15
    SRCNN : 2014, 30.09
    VDSR : 2016, 31.35
    EDSR : 2017, 32.46
    RCAN : 2018, 32.63
    SwinIR : 2021, 32.92
    DiffIR : 2023, 33.41
```

```mermaid
graph LR
    A[CNN<br>局部特征] --> D[混合架构]
    B[Transformer<br>全局建模] --> D
    C[Diffusion<br>概率生成] --> D
    D --> E1[EfficientSR<br>移动端]
    D --> E2[Text2SR<br>语义引导]
    D --> E3[3D-SR<br>体积重建]
    
    style D fill:#f9f,stroke:#333
```

## LHAI模型命名规范

模型命名规则格式:

```
[模型架构]_[实验代号]_[训练轮数]epo_[批大小]bth_[数据集简写]_[版本].pth
```

字段说明:

| 字段名   | 示例                           | 含义说明                         |
| ----- | ---------------------------- | ---------------------------- |
| 模型架构  | `CNN` / `DIFFUSION` / `ResNet`    | 网络模型名称，便于快速识别所用架构            |
| 实验代号  | `EXP01`                    | 训练实验的编号或标签，用于区分不同实验          |
| 训练轮数  | `400epo`                     | 训练的 epoch 数，便于识别训练时长         |
| 批大小   | `32bth`                      | 批次大小，表示一次训练迭代中使用的样本数量        |
| 数据集简写 | `poisson`            | 训练数据集文件名简写，便于快速关联数据集         |
| 版本号   | `v1` / `v2`                  | 模型版本号，便于模型迭代更新及版本管理（可省略）          |

示例:

```
CNN_EXP01_400epo_32bth_poisson.pth
```

模型存储方式介绍:

> 本模型以 PyTorch 的 `.pth` 格式存储，包含训练好的权重参数，命名规则反映了模型的架构、训练配置与对应数据集，方便版本管理和快速定位。

在代码配置中的规范写法示例:

```python linenums="1"
TRAIN_EXP_NAME        = "EXP_0_1"
TRAIN_MODEL_NAME      = "CNN"
TRAIN_MODEL_PY        = PROJ_ROOT / "LHAI" / "models" / f"{TRAIN_MODEL_NAME}_{TRAIN_EXP_NAME}.py"
TRAIN_DATA_DIR        = PROJ_ROOT / "data" / "POISSON"
TRAIN_DATA_NAME       = "poisson_src_bkg.pkl.npy"
TRAIN_DATA_PATH       = TRAIN_DATA_DIR / TRAIN_DATA_NAME
TRAIN_SEED            = 0
TRAIN_TYPE            = "poissonsrc+bkg_highresorig"
TRAIN_FRAC            = 0.8
TRAIN_EPOCHS          = 400
TRAIN_BATCH_SIZE      = 32
TRAIN_LATENT_DIM      = 64
TRAIN_LR_MAX          = 5e-4
TRAIN_LR_MIN          = 5e-6
```