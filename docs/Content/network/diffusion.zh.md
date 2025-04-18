# diffusion 实验文档

## 1 实验资料

1. sr3-notebook.ipynb 文件：谷歌的sr3（超分辨）模型

2. reshift项目复现：开源在github上的diffusion超分辨项目

3. Diffusion models for super-resolution microscopy: a tutorial 论文

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