# 数据介绍与规范化标准

## 1 数据集命名规则与存储方式

数据集命名规则：

```
[源类型]_[数据数量]_[图片大小]_[用途]_[预处理]_[版本].npy
```

数据集存储方式介绍：


> 本数据集以 NumPy 数组（.npy）格式存储，数据结构为四维数组 (N, 2, H, W)，其中 N 表示样本数量，H 和 W 分别为图像的高度与宽度。每个样本包含一对图像，按顺序存储为清晰图像与模糊图像，即维度索引为 0 的图像为清晰图（如高分辨率或参考图），索引为 1 的图像为对应的模糊图（如低分辨率或退化图）。

数据集命名规则说明

| 字段名    | 示例                       | 含义说明                                               |
| ------ | ------------------------ | -------------------------------------------------- |
| `源类型`  | `mic` / `sim` / `real`   | 数据来源类型：如 `mic` 表示显微图像，`sim` 表示模拟数据，`real` 表示真实采集图像 |
| `数据数量` | `7000` / `12000`         | 当前文件中包含的图像对数量，便于快速了解数据规模                           |
| `图片大小` | `64` / `128`      | 单张图像的空间尺寸（高 × 宽），适用于模型输入配置                         |
| `用途`   | `train` / `val` / `orig` | 数据用途标记：训练集、验证集或原始数据集                                 |
| `预处理`  | `norm` / `crop` / `none` | 数据是否经过归一化、裁剪等预处理操作，若无预处理可省略             |
| `版本`   | `v1` / `v2.1` / `v3`     | 数据集版本号，支持迭代维护与更新追踪                                 |



## 2 LHAI数据集与介绍

原始数据集：

| 文件名                                    | 含义说明                                         |      创建时间      |
| -------------------------------------- | -------------------------------------------- |------- |
| `miragesearch_4200_140_orig.npy`     | 老师那边最新的一组原始模拟数据 |        |

训练集（以创建时间排序）：

| 文件名                                    | 含义说明                                         |创建时间      |
| -------------------------------------- | -------------------------------------------- |------ |
| `biosr_41190_64_train_crop.npy`     | 显微图像来源，纤毛 |   |
| `cilium_6512_64_train.npy` | 显微图像来源，荧光小球      |   |
| `cube_1504_64_train.npy`      | 显微图像来源2，荧光小球             |   |
| `pysted_5204_64_train.npy`      | 显微图像来源，量子点             |   |
| `tangxiao_7000_64_train.npy`      | 唐晓模拟数据集             |   |
| `halos_1200_64_train_crop.npy`      | 交大模拟数据集             |   |
| `xingwei_10000_64_train_v1.npy`      | 星维模拟4表征数据集            |   |


测试集：

| 文件名                                    | 含义说明                                         |创建时间      |
| -------------------------------------- | -------------------------------------------- |------ |
| `nhit100_1_64_val.npy`     | 一张2源的简单测试图片，E=100 |   |
| `nhit2000_1_64_val.npy` | 一张2源的简单测试图片，E=2000      |   |

## 3 数据集相关代码

### LHAI Dataset模块

归一化为[0,1]：

```
import torch
from tifffile import tifffile
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split, ConcatDataset

class ImageDataset(Dataset):
    def __init__(self, num_to_learn, path_data,inverse=False):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []

        if not os.path.exists(path_data):
            raise FileNotFoundError("Blurry or Original data file not found.")
        datas = np.load(path_data,allow_pickle=True)#.astype(np.object)
        blurry_datas = np.stack(datas[:,1])
        original_datas = np.stack(datas[:,0])

        if inverse == False:
            idx_beg = 0;
            idx_end = num_to_learn;
        else:
            idx_beg = blurry_datas.shape[0]-num_to_learn;
            idx_end = blurry_datas.shape[0];

        for i in range(idx_beg,idx_end):
            blurry_data = blurry_datas[i]
            original_data = original_datas[i]
            
            img_blurry = (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min())
            img_original = (original_data - original_data.min()) / (original_data.max() - original_data.min())
            
            img_blurry = Image.fromarray(img_blurry)
            img_original = Image.fromarray(img_original)
        
            img_blurry = self.transform(img_blurry)
            img_original = self.transform(img_original)
            
            self.data.append((img_blurry, img_original))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
```

归一化为[-1,1]：

```
import torch
from tifffile import tifffile
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split, ConcatDataset

class ImageDataset(Dataset):
    def __init__(self, num_to_learn, path_data,inverse=False):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data = []

        if not os.path.exists(path_data):
            raise FileNotFoundError("Blurry or Original data file not found.")
        datas = np.load(path_data,allow_pickle=True)#.astype(np.object)
        blurry_datas = np.stack(datas[:,1])
        original_datas = np.stack(datas[:,0])

        if inverse == False:
            idx_beg = 0;
            idx_end = num_to_learn;
        else:
            idx_beg = blurry_datas.shape[0]-num_to_learn;
            idx_end = blurry_datas.shape[0];

        for i in range(idx_beg,idx_end):
            blurry_data = blurry_datas[i]
            original_data = original_datas[i]
            
            img_blurry = 2 * (blurry_data - blurry_data.min()) / (blurry_data.max() - blurry_data.min()) - 1
            img_original = 2 * (original_data - original_data.min()) / (original_data.max() - original_data.min()) - 1

            #img_blurry = blurry_data/blurry_data.max()
            #img_original = original_data/original_data.max()
            
            img_blurry = Image.fromarray(img_blurry)
            img_original = Image.fromarray(img_original)
        
            img_blurry = self.transform(img_blurry)
            img_original = self.transform(img_original)
            
            self.data.append((img_blurry, img_original))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
```

### 数据集规范化预处理、检查模块

规范化预处理流程：图像尺寸统一化 - 数据集灰度化 - 归一化处理 - 数据集扩张 - 数据增强 - 重估数据/低信息数据清理

检查模块：（暂定）


### 其他代码附件

展示数据集大小、imshow部分数据：[imshow 点击下载](daia_imshow.ipynb)

由jpg等主流数据存储方式转换为lhai数据存储方式：（代码丢失）

将两个lhai数据集随机混合起来：（代码丢失）

将大于64尺寸的代码裁剪为64：代码丢失
