# diffusion 实验文档

## 1 实验资料

- [ ] sr3-notebook.ipynb 文件：谷歌的sr3（超分辨）模型

- [ ] reshift项目复现：开源在github上的diffusion超分辨项目

- [x] Diffusion models for super-resolution microscopy: a tutorial 论文

## 2 论文复现

完成论文notebook复现。

!!! 问题 1

    在这篇论文的框架里面，有没有对输入输出图片尺寸的规定？


要将输入数据尺寸从 `128×128` 修改为 `64×64`，**不需要修改 UNet 模型结构**，但需要调整以下代码部分：


### 1. **修改 `Diffusion` 类的 `img_size` 参数**

在初始化扩散模型时，显式指定 `img_size=64`：

```python
# 修改前
diffusion = Diffusion(noise_steps=201, beta_start=0.0001, beta_end=0.02)

# 修改后
diffusion = Diffusion(
    noise_steps=201,
    beta_start=0.0001,
    beta_end=0.02,
    img_size=64  # 关键调整
)
```

### 2. **验证 UNet 结构的兼容性**

当前 `AttentionUNet` 的参数为 `channels=[32, 64, 128]`，表示有 **3 次下采样**（每次分辨率减半），因此输入尺寸需满足：

\[
\text{输入尺寸} \div 2^3 = \text{整数}
\]

对于 `64×64`：

\[
64 \div 8 = 8 \quad (\text{合法})
\]

因此，UNet 完全兼容 `64×64` 的输入，无需调整其结构。

### 3. **调整数据预处理**

确保训练数据集的图像被正确缩放到 `64×64`。例如：

```python
# 数据加载时需确保图像尺寸为 64×64
low_res_image, high_res_image = train_dataset[...]  # 假设数据已预处理为 64×64
```

### 4. **反向扩散过程的兼容性**

在 `reverse_diffusion` 方法中，噪声初始化已依赖 `self.img_size`，因此自动适配 `64×64`：

```python
x = torch.randn((n_images, n_channels, self.img_size, self.img_size))  # 自动生成 64×64 的噪声
```

### 5. **潜在注意事项**

- **UNet 特征图尺寸**：  

  输入 `64×64` 时，经过 3 次下采样后，最深层特征图尺寸为：

  \[
  64 \div 2^3 = 8 \times 8
  \]

  若特征图过小（如低于 `4×4`），可能影响模型性能，但 `8×8` 是合理的。

- **条件输入的一致性**：  

  如果任务是多模态输入（如低分辨率图像作为条件），需确保条件图像的尺寸也为 `64×64`。

### 总结

- **无需修改 UNet 结构**，只需调整 `Diffusion` 的 `img_size=64` 和数据预处理。

- **完整修改步骤**：

  1. 在 `Diffusion` 初始化时设置 `img_size=64`。

  2. 确保训练数据为 `64×64`。

  3. 其他代码（如 UNet 定义、扩散过程）保持不变。

若代码中未硬编码 `128×128`（如可视化部分），则无需额外修改。建议运行后检查张量形状是否匹配（如 `print(x.shape)`）。

!!! 问题2

    数据的标准化normalize部分

!!! 问题3

    训练数据数量级问题，至少需要提高一个数量级

## 3 novel技术路线：扩散模型并不负责“完整生成图像”，而是“微调/修复/增强”已有图像

方法描述：用已有 UNet/CNN 超分模型（比如 sr_image = my_model(lr_image)）输出一个粗略的 HR 图像。然后我们接着使用一个训练好的或待训练的 diffusion_model，进一步细化这个图。两种路线：

1. 选择已有的与训练diffusion model

2. 自己训练专用的diffusion reinforcement model

### 问题1 我将原来的 UNet 输出结果作为 Diffusion 的输入，这样是“串联两个模型”，真的更好吗？

1. 加速diffusion推理过程  
  原始 diffusion 模型从纯噪声生成图像，推理要走几十甚至上百步；  
  而 SR 模型已经提供了一个合理结构，diffusion 只需“少量步骤”完成 refinement → 大大加速  

2. 融合结构感知 + 细节恢复  
  UNet 结构善于学习空间结构，但细节能力差；  
  diffusion 善于采样自然细节纹理 → 两者互补；  
  有点像先画草图再上色润色。  

3. 模块复用  
  如果你原模型已经训练得不错，就无需推倒重来，只需训练个小 diffusion refine 网络即可。

### 问题2 相关研究与论文

| 名称 | 内容概述 | 论文链接 |
|------|-----------|----------|
| **SR3 (Super-Resolution via Repeated Refinement)** | Google 提出的 diffusion 超分方法，逐步 refine，效果显著 | [arXiv:2104.07636](https://arxiv.org/abs/2104.07636) |
| **SDEdit (Structure-preserving Diffusion Editing)** | 使用已有图像作为起点，然后用扩散 refine，适用于 image2image task | [arXiv:2108.01073](https://arxiv.org/abs/2108.01073) |
| **DDPM-SR (Janspiry)** | SR from coarse to fine，通过一个 baseline 模型 + DDPM 精细化 | [GitHub 项目](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) |
| **IR-SDE** | 学习 residual image，扩散只预测细节部分 | [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) |


## 4 edge技术路线

路线1（目前最新）Diffusion Transformer 架构（DiT 系列）

近年来超分辨率（Super-Resolution, SR）领域的研究重心已经从传统的 UNet + Diffusion 组合，逐步转向了更先进的 **Transformer 架构与 Diffusion 模型的融合**。（可能不准确）

### 1. **Diffusion Transformer 架构（DiT 系列）**

#### ✅ DiT-SR（2024）

- **核心思想**：将 Transformer 融入扩散模型，采用 U-shaped 架构，统一各阶段的 Transformer 块设计。
- **创新点**：提出频率自适应的时间步调制模块，增强模型处理不同频率信息的能力。
- **性能表现**：在多个基准数据集上，DiT-SR 显著优于现有的从头训练的扩散超分方法，甚至超过了一些基于预训练 Stable Diffusion 的方法。

#### ✅ DiT4SR（2025）

- **核心思想**：针对真实世界图像超分问题，提出了 DiT4SR 模型。
- **创新点**：将低分辨率图像的嵌入集成到 DiT 的注意力机制中，实现低分辨率和生成特征之间的双向信息流动。
- **性能表现**：通过这种设计，DiT4SR 在真实图像超分任务中表现出色，优于传统的 UNet 架构。

### 2. **Residual Diffusion 架构**

#### ✅ ResDiff（2023）

- **核心思想**：结合 CNN 和扩散模型，CNN 负责恢复低频信息，扩散模型预测残差（高频细节）。
- **创新点**：在频率域中引导扩散过程，专注于高频细节的恢复。
- **性能表现**：在多个基准数据集上，ResDiff 在生成质量和多样性方面优于先前的扩散方法。

### 3. **高效扩散模型**

#### ✅ ResShift（NeurIPS 2023）

- **核心思想**：通过残差转移机制，构建高效的扩散模型，显著减少采样步骤。
- **创新点**：设计了灵活控制转移速度和噪声强度的噪声调度策略。
- **性能表现**：在合成和真实世界数据集上，仅用 20 个采样步骤即可获得优异或至少可比的性能。


### 总结

| 方案类型               | 优势                                                         | 代表模型             |
|------------------------|--------------------------------------------------------------|----------------------|
| Transformer + Diffusion | 更强的建模能力，处理复杂结构和细节信息                     | DiT-SR, DiT4SR       |
| Residual Diffusion     | 专注于高频细节恢复，提升图像质量                           | ResDiff              |
| 高效扩散模型           | 减少采样步骤，加速推理过程，适用于实时应用                 | ResShift             |

## 5 回顾命题

### 不同技术路线的比较

| 模型类型                   | 代表模型       | PSNR 提升（对比 CNN 基准） | 特点                     |
|----------------------------|----------------|-----------------------------|--------------------------|
| **传统 CNN/UNet**         | SRCNN, EDSR, UNetSR | baseline                   | 高速、易训练             |
| **Residual GAN**           | ESRGAN, Real-ESRGAN | +0.5~1.0 dB                | 细节增强、有时会假象     |
| **Diffusion**              | SR3, DDNM       | +1.5~2.5 dB                  | 细节拟真度高、采样慢     |
| **Residual Diffusion**     | ResDiff         | +2.0~3.0 dB                  | 高频细节修复效果出色     |
| **Diffusion + Transformer**| DiT-SR, DiT4SR  | +2.5~4.0 dB                  | 全局结构建模能力强       |

📌 **说明**：以上提升是针对 `DIV2K`, `RealSR`, `BSD100` 等 benchmark 数据集上的平均情况，具体提升程度依数据集、任务难度和噪声情况而异。

从上面可以看出，最先进的的模型相对于传统CNN，提升也不大。因此现在需要回顾一下最初命题，进行详细讨论。

### 任务特征分析

| 特征项             | 描述                                                                 |
|--------------------|----------------------------------------------------------------------|
| 输入输出           | 配对的 LR-HR 天文图像对                                              |
| 图像特性           | 几乎无复杂场景变化、但细节结构极其丰富                               |
| 可接受推理速度     | 接受度高（如果用于研究，不一定追求实时）                                 |
| 对“伪影/假象”容忍度 | 极低（科学图像不能“编造”细节）                                        |
| 对边缘/纹理敏感度   | 极高（细节往往决定诊断或分析结果）                                    |

> ✅ **结论**：需要一套“结构保守 + 高频增强 + 无假象”的方案

### 技术路线：**Residual Diffusion for Microscopy Super-Resolution**

我们选用 **ResDiff**（残差扩散）架构，结合你现有的 CNN/UNet 网络，组成如下技术管线：

#### 🔹 Step 1：训练一个传统 UNet/CNN 网络
- 输入：LR 显微图像
- 输出：初步还原的 HR 图像（结构大致准确但细节有限）
- 损失函数：MSE + Perceptual Loss（可选 SSIM Loss）

#### 🔹 Step 2：构建 Residual Diffusion 模块
- 输入：Step 1 的输出图像（作为 conditional input）
- 目标输出：Ground truth HR 图像
- 训练目标：预测高频残差（即真实 HR 与 Step 1 输出图像之间的差）
- 网络结构：使用 DDPM 或 Denoising Network + Timestep Embedding（可使用基于 U-Net or Transformer 的 block）

#### 🔹 Step 3：组合推理

```text
LR 显微图像 --> UNet 预测初稿 --> Residual Diffusion 细节补全 --> 最终 HR
```

#### 优点
- **结构安全性强**：初稿结构由传统网络给出，避免“虚构结构”
- **细节真实细腻**：高频部分由 diffusion 精细补充
- **控制能力强**：可以对 diffusion 进行结构感知或 frequency guidance，避免 hallucination


| 任务 | GitHub 项目 | 简介 |
|------|-------------|------|
| Diffusion 超分 | [`tangke03/DDNM`](https://github.com/tangke03/DDNM) | Residual Diffusion 支持图像复原 |
| Microscopy SR | [`csbDeep`](https://github.com/CSBDeep/CSBDeep) | 用于显微图像的 DL-SR 平台，结构类似 UNet |
| Stable Diffusion SR | [`CompVis/latent-diffusion`](https://github.com/CompVis/latent-diffusion) | 支持用低维 latent space 进行超分 |

​是的，您提到的“Residual Diffusion for Microscopy Super-Resolution”技术路线在当前的研究中已有相关的论文和项目实现。以下是与该技术路线相关的主要论文和开源项目：

---

### 📄 相关论文

1. **ResDiff: Combining CNN and Diffusion Model for Image Super-Resolution**  
  **作者**：Shuyao Shang 等  
  **摘要** 该论文提出了一种名为 ResDiff 的新型扩散概率模型，结合了卷积神经网络（CNN）和扩散模型的优势，用于单幅图像超分辨率任务。作者认为，直接使用扩散模型进行图像超分辨率可能效率低下，因为简单的 CNN 就能恢复主要的低频内容。因此，ResDiff 利用 CNN 先恢复低频信息，再通过扩散模型预测高频残差，从而提高了效率和性能

1. **HF-ResDiff: High-Frequency-Guided Residual Diffusion for Multi-dose PET Reconstruction**  
  该论文提出了一种高频引导的残差扩散模型（HF-ResDiff），用于多剂量正电子发射断层扫描（PET）图像的重建该方法通过高频信息引导扩散过程，有效地保留了图像的高频细节，提升了图像质量

总结

| 项目              | 推荐选择                   | 理由                                                   |
|-------------------|----------------------------|--------------------------------------------------------|
| 网络结构           | UNet + Residual Diffusion  | 结构安全 + 高频增强 + 抑制伪影                        |
| 损失函数           | MSE + SSIM + LPIPS         | 平衡像素级准确度与感知质量                            |
| 训练策略           | 分阶段训练（先 CNN 后扩散）| 加快收敛，便于调试                                     |
| 推理部署           | 合并模型/模块级组合        | 根据硬件部署条件决定                                   |
