# LHAI 函数模块

这部分是LHAI代码的核心

**这部分的中央思想是：在保持LOSS\DATA\TRAIN...函数不变的情况下，控制变量，改变神经网络，消除玄学影响，增加训练效率**

因此，这部分是独立于神经网络之外的可调参函数。按照重要性排序介绍：

## Dataset

该模块负责数据加载与预处理，是训练流程的入口。

```python
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, num_to_learn, path_data, inverse=False, data_range=1.0):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []

        if not os.path.exists(path_data):
            raise FileNotFoundError("Blurry or Original data file not found.")

        datas = np.load(path_data, allow_pickle=True)
        blurry_datas = np.stack(datas[:, 1])
        original_datas = np.stack(datas[:, 0])

        if not inverse:
            idx_beg = 0
            idx_end = num_to_learn
        else:
            idx_beg = blurry_datas.shape[0] - num_to_learn
            idx_end = blurry_datas.shape[0]

        for i in range(idx_beg, idx_end):
            blurry_data = blurry_datas[i]
            original_data = original_datas[i]

            # 数据归一化处理
            if abs(data_range - 1.0) < 1e-5:
                img_blurry = (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min())
                img_original = (original_data - original_data.min()) / (original_data.max() - original_data.min())
            elif abs(data_range - 2.0) < 1e-5:
                img_blurry = 2 * (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min()) - 1
                img_original = 2 * (original_data - original_data.min()) / (original_data.max() - original_data.min()) - 1
            else:
                raise ValueError("datarange must be 1.0 or 2.0")

            # 转为PIL Image，方便后续转换为Tensor
            img_blurry = Image.fromarray(img_blurry)
            img_original = Image.fromarray(img_original)

            # 转Tensor
            img_blurry = self.transform(img_blurry)
            img_original = self.transform(img_original)

            self.data.append((img_blurry, img_original))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
```


* 读取 `path_data` 指定的 `.npy` 文件，数据格式为 `[ [original, blurry], ... ]`，两个图像均为numpy数组。
* 支持选择是否从数据尾部（inverse=True）读取样本。
* 支持两种归一化范围：

  * `[0,1]` 归一化（data\_range=1.0）
  * `[-1,1]` 归一化（data\_range=2.0）
* 采用PIL转换为Tensor，保持与PyTorch训练流程兼容。
* 设计简单易扩展，方便添加额外预处理或数据增强。

## Train

### 1. `train`

- **核心功能**：普通神经网络训练流程
- **特点**：
  - 动态学习率调度（scheduler）
  - 训练与测试阶段分离，均计算平均损失
  - 日志实时记录（logger + 自定义log函数）
  - 训练过程损失数据实时存储（LOSS_PLOT等列表）
- **核心步骤**：
  1. 清空日志文件，写入初始训练信息
  2. 进入epoch循环，执行训练：
     - 模型设为train模式
     - 遍历训练集，前向计算、反向传播、优化
     - 累计训练损失，计算均值
  3. 切换测试模式，遍历测试集，计算测试损失均值
  4. 记录日志，更新绘图列表，执行学习率调度

### 2. `train_diffusion`

- **核心功能**：扩散模型（Diffusion Model）训练流程
- **特点**：
  - 结合位置编码、噪声步数等扩散模型特有参数
  - 训练中使用自定义`prepare_data`处理输入和噪声
  - 支持多种评估指标（MAE、MS-SSIM、SSIM、PSNR、NRMSE）计算和记录
  - 训练和评估阶段分离，支持动态日志和时间统计
  - 学习率调度器调用在epoch末尾
- **核心步骤**：
  1. 初始化日志文件和日志信息
  2. epoch循环：
     - 训练模式：
       - 遍历训练批次，准备数据（加噪等），前向计算损失，反向传播，优化
       - 定期打印训练损失
     - 评估模式：
       - 遍历测试批次，利用`diffusion.reverse_diffusion`生成图像
       - 若开启指标计算，计算并累积指标
     - 记录当前epoch的训练损失与耗时日志
     - 指标按批次数归一化并保存
  3. 更新学习率

---

设计理念：

- **解耦复杂性**：训练与评估模块职责分明，指标计算可按需开启
- **动态控制**：学习率随训练进度自动调节，提升收敛效率
- **易扩展性**：支持多指标评估，方便对模型性能全面监控
- **日志与可视化**：日志与绘图列表实时更新，方便训练监控与结果分析

---

使用提示：

- 训练时注意准备好对应的数据加载器`dataloader`和`trainloader`、`testloader`
- `train_diffusion`需要配合扩散模型结构和预处理函数`prepare_data`使用
- 评估指标函数需提前定义并传入，避免运行时错误
- 训练日志文件路径`logpath`应确保可写权限


## LOSS

### 1. SSIM 相关

- **`ssim_function(img1, img2, window_size=11, data_range=255.0, sigma=1.5)`**  
  经典结构相似性指标计算，基于高斯滤波与滑动窗口卷积，返回两个图像的平均SSIM值。  
  内部调用`gaussian`生成高斯权重窗。  
  输入为二维numpy数组。

- **`batch_ssim(img1, img2)`**  
  批量SSIM计算，输入为Tensor，逐张转PIL后调用`ssim_function`，最终返回批次平均值。  
  适用于批处理的训练评估。

### 2. MSE 与 PSNR

- **`batch_psnr(img1, img2, max_val=255.0)`**  
  批量峰值信噪比计算，基于均方误差MSE转换而来，适合衡量图像重建质量。

### 3. 自定义复合损失类

- **`Custom_criterion(nn.Module)`**  
  结合MSE和SSIM的加权损失。  
  - `mse_weight`与`ssim_weight`控制权重。  
  - `forward`函数返回加权后SSIM损失（示例代码中仅返回了ssim_loss，用户可根据注释调整）。

### 4. JS散度（Jensen-Shannon Divergence）

- **`jsdiv(img1, img2)`**  
  计算两个张量的JS散度，先做softmax归一化，再计算信息熵相关项，返回平均JS散度。  
  适合测量概率分布差异。

- **`jsdiv_single(img1, img2)`**  
  类似`jsdiv`，但返回单个样本的散度值列表，不做平均。

### 5. 其他

- 注释部分标记了多种常用损失（L1、L2、MAE、RMSE、VGG损失、交叉熵、Dice等），可根据需求扩展。

### 6. 组合示例

- **`msejsloss(img1, img2)`**  
  MSE与JS散度的加权组合，示例权重为0.2与0.8，体现复合损失的灵活性。

---

设计理念

- **模块化**：单一功能函数拆分，方便替换和调试
- **兼容性**：输入支持numpy数组与PyTorch张量，适应训练和评估需求
- **灵活配置**：权重可调，支持多种指标融合
- **效率考量**：批量操作和GPU支持，兼顾性能与精度

---

使用建议

- SSIM相关函数适合质量评估但计算开销较大，训练时可按需调用
- JS散度适合概率分布对比，常用于生成模型或分类任务
- 复合损失`Custom_criterion`可作为基础，用户可根据任务需求调整权重及组合方式
- 保证输入数据范围和格式符合预期（如`data_range`，Tensor的形状和类型）

## dataprocess

构造函数中，传入参数包括样本数量、数据路径、是否反向加载、归一化范围、去重阈值、数据增强开关及方式、伽马增强、低信息样本过滤等，支持灵活定制数据处理流程。

数据首先安全读取为numpy数组，避免`object`类型，随后拆分模糊图与原图两部分。

针对不同需求，支持反向加载样本区间。对每张图像执行归一化处理，支持标准[0,1]或[-1,1]两种数据范围。

后续根据配置依次执行：

- **低信息样本移除**：按原图均值判断，剔除信息量过低样本，防止训练干扰。
- **重复样本去重**：基于GPU加速的批量SSIM计算，对模糊图像去重，显著降低样本冗余。
- **数据增强**：支持多角度旋转扩增，采用矢量化实现高效处理。
- **伽马增强**：对原图做伽马校正，提升图像对比度和细节表现。

所有处理后数据转换为PyTorch张量格式 `(1, H, W)`，便于后续训练。

---

核心私有方法：

- `_normalize`：根据传入的`data_range`执行归一化，确保像素值符合模型预期。
- `_remove_low_info`：根据阈值过滤均值过低的原图样本。
- `_remove_duplicates_batch_gpu`：利用`torchmetrics`的SSIM指标批量计算相似度，实现GPU加速去重。
- `_augment_vectorized`：批量旋转多角度数据增强，扩充样本空间。
- `_enhance_gamma_vectorized`：统一对原图执行伽马变换，提升图像动态范围。

---

设计要点：

- 严格保证数据格式统一和计算效率，适配大规模数据处理。
- 充分利用GPU加速，减少重复数据计算开销。
- 支持多阶段处理流程，可按需开启关闭，灵活应对不同训练策略。
- 最终输出符合深度学习框架标准的Tensor格式，方便模型直接调用。

## Log

函数`log(logpath, message)`：

- 接收日志文件路径`logpath`和字符串`message`。
- 以追加模式打开日志文件，自动处理编码为UTF-8。
- 将消息写入文件，并自动换行。

---

设计理念：

- 极简实现，方便在训练流程中多处调用。
- 保障日志连续性和完整性，不覆盖已有内容。
- 保持编码一致，避免中文等特殊字符写入异常。