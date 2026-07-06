# DATAGENERATING

## 一、本征数据集的生成

本节记录各类本征形态数据集生成脚本的用途与基本用法。各脚本通常直接用 `python <脚本名>.py` 运行：不加 `--generate-dataset` 时预览单个样本，加上 `--generate-dataset` 时批量生成 `.npy` 数据集。

### 1. `Type_A_COMPACT_generating.py`

`Type_A_COMPACT_generating.py` 用于生成 Type A 中的紧致源/点源本征数据集，模拟未分辨或近似未分辨的伽马射线源形态。脚本支持两类形态：`point` 将全部流量放在最近像素上，表示理想点源；`compact` 使用窄二维圆对称高斯分布，表示略有展宽的紧致源。

批量生成时，输出为形状 `(样本数, 图像尺寸, 图像尺寸)` 的 `float32` 数组，并保存为 `.npy` 文件。默认不加 `--random` 时生成固定参数的纯 compact 数据集；加上 `--random` 后，每个样本会按 `--point-fraction` 的概率生成 pointlike，否则生成 compact，并从 `--intensity-min/--intensity-max` 与 `--sigma-min/--sigma-max` 指定范围内采样流量和 compact 宽度。

常用生成命令示例：

```bash
python Type_A_COMPACT_generating.py --generate-dataset --dataset-count 1000 --random --point-fraction 0.5 --intensity-min 1 --intensity-max 100 --sigma-min 0.01 --sigma-max 0.1 --seed 0 --dataset-output TYPEA_COMPACT_1000_128_GT.npy
```

主要参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--generate-dataset` | 启用批量数据集生成模式；不加时只预览单个样本 | 关闭 |
| `--dataset-count` | 批量生成的样本数 | `100` |
| `--dataset-output` | 输出 `.npy` 文件路径 | `Type_A_Compact_100_128_GT.npy` |
| `--random` | 随机生成 point/compact，并随机采样流量和宽度；批量生成和单图预览均生效 | 关闭 |
| `--point-fraction` | `--random` 模式下生成 pointlike 样本的概率 | `0.5` |
| `--source-kind` | 非随机单图预览时的源类型，可选 `point` 或 `compact` | `compact` |
| `--intensity` | 非随机模式下的总流量 | `10.0` |
| `--intensity-min` / `--intensity-max` | `--random` 模式下总流量的对数均匀采样范围 | `1.0` / `100.0` |
| `--sigma` | 非随机 compact 源的高斯标准差，单位为度 | `0.05` |
| `--sigma-min` / `--sigma-max` | `--random` 模式下 compact 源高斯标准差的采样范围，单位为度 | `0.01` / `0.1` |
| `--center-mu` / `--center-sigma` | 源中心位置高斯分布的均值和标准差，单位为度 | `0.0` / `1.5` |
| `--max-offset` | 源中心相对视场中心的最大偏移，单位为度 | `3.2` |
| `--size` | 输出图像边长像素数 | `128` |
| `--fov` | 视场大小，单位为度 | `6.4` |
| `--seed` | 随机种子，用于复现实验 | `None` |
| `--save` | 单图预览的保存路径 | 不保存 |

脚本运行结束后会打印保存文件名、数组形状、数据类型、像素最小/最大值，以及每个样本总流量的最小值、最大值和均值。建议生成正式数据集时显式指定 `--dataset-output`，避免覆盖已有文件。

### 2. `Type_A_GAUSSIAN_generating.py`

`Type_A_GAUSSIAN_generating.py` 用于生成 Type A 中的扩展高斯类本征数据集，将高斯源拆分为四个层级：`perfect` 是纯圆对称高斯，只由位置、总流量和 `sigma` 控制；`elliptical` 在此基础上加入 `axis-ratio` 和方向角；`perftextured` 在圆对称高斯上叠加头尾调制和分形湍动纹理；`elliptextured` 则在椭圆高斯上叠加同样的真实纹理效果。

批量生成时，输出为形状 `(样本数, 图像尺寸, 图像尺寸)` 的 `float32` 数组，并保存为 `.npy` 文件。默认不加 `--random` 时生成固定参数的纯 `perfect` 数据集；加上 `--random` 后，每个样本会按 `--perfect-fraction`、`--elliptical-fraction`、`--perftextured-fraction`、`--elliptextured-fraction` 的相对比例抽取类型，并从对应的 `min/max` 参数范围内采样流量、宽度、轴比、调制和湍动强度。

常用生成命令示例：

```bash
python Type_A_GAUSSIAN_generating.py --generate-dataset --dataset-count 1000 --random --perfect-fraction 1 --elliptical-fraction 1 --perftextured-fraction 1 --elliptextured-fraction 1 --intensity-min 1 --intensity-max 100 --sigma-min 0.1 --sigma-max 0.4 --axis-ratio-min 0.45 --axis-ratio-max 1.0 --compression-min 0.35 --compression-max 0.9 --tail-strength-min 0.5 --tail-strength-max 2.0 --transition-min 0.05 --transition-max 0.25 --turbulence-alpha-min 0.05 --turbulence-alpha-max 0.3 --seed 0 --dataset-output Type_A_GAUSSIAN_1000_128_GT.npy
```

主要参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--generate-dataset` | 启用批量数据集生成模式；不加时只预览单个样本 | 关闭 |
| `--dataset-count` | 批量生成的样本数 | `100` |
| `--dataset-output` | 输出 `.npy` 文件路径；建议显式指定 Type A Gaussian 文件名 | `Type_A_Compact_100_128_GT.npy` |
| `--source-kind` | 非随机单图预览时的高斯类型，可选 `perfect`、`elliptical`、`perftextured`、`elliptextured` | `perfect` |
| `--random` | 随机抽取高斯类型，并随机采样物理参数；批量生成和单图预览均生效 | 关闭 |
| `--perfect-fraction` | `--random` 模式下 `perfect` 类型的相对比例 | `0.0` |
| `--elliptical-fraction` | `--random` 模式下 `elliptical` 类型的相对比例 | `0.0` |
| `--perftextured-fraction` | `--random` 模式下 `perftextured` 类型的相对比例 | `0.0` |
| `--elliptextured-fraction` | `--random` 模式下 `elliptextured` 类型的相对比例 | `1.0` |
| `--intensity` | 非随机模式下的总流量 | `10.0` |
| `--intensity-min` / `--intensity-max` | `--random` 模式下总流量的对数均匀采样范围 | `1.0` / `100.0` |
| `--sigma` | 非随机模式下的高斯标准差，单位为度 | `0.1` |
| `--sigma-min` / `--sigma-max` | `--random` 模式下高斯标准差的采样范围，单位为度 | `0.1` / `0.4` |
| `--axis-ratio` | 非随机 `elliptical` / `elliptextured` 的短轴/长轴比例 | `0.7` |
| `--axis-ratio-min` / `--axis-ratio-max` | `--random` 模式下轴比采样范围 | `0.45` / `1.0` |
| `--compression-min` / `--compression-max` | `perftextured` / `elliptextured` 的头部压缩调制采样范围 | `0.35` / `0.9` |
| `--tail-strength-min` / `--tail-strength-max` | `perftextured` / `elliptextured` 的尾部衰减尺度采样范围，单位为度 | `0.5` / `2.0` |
| `--transition-min` / `--transition-max` | `perftextured` / `elliptextured` 的头尾过渡宽度采样范围，单位为度 | `0.05` / `0.25` |
| `--turbulence-alpha-min` / `--turbulence-alpha-max` | `perftextured` / `elliptextured` 的湍动调制强度采样范围 | `0.05` / `0.3` |
| `--turbulence-beta` | 分形湍动噪声的谱指数 | `3.0` |
| `--center-mu` / `--center-sigma` | 源中心位置高斯分布的均值和标准差，单位为度 | `0.0` / `1.0` |
| `--max-offset` | 源中心相对视场中心的最大偏移，单位为度 | `1.6` |
| `--size` | 输出图像边长像素数 | `128` |
| `--fov` | 视场大小，单位为度 | `6.4` |
| `--seed` | 随机种子，用于复现实验 | `None` |
| `--save` | 单图预览的保存路径 | 不保存 |

脚本运行结束后会打印保存文件名、数组形状、数据类型、像素最小/最大值，以及每个样本总流量的最小值、最大值和均值。

### 3. `Type_A_SHELL_generating.py`

`Type_A_SHELL_generating.py` 用于生成 Type A 中的壳状本征数据集，模拟类似超新星遗迹的中空、环状、碎片化壳层结构。脚本先生成随机噪声并进行平滑，再通过阈值门控得到不连续的壳层碎片；随后叠加径向环形亮度分布和大尺度方位不对称，得到破碎、非均匀的 shell morphology。

批量生成时，输出为形状 `(样本数, 图像尺寸, 图像尺寸)` 的 `float32` 数组，并保存为 `.npy` 文件。每个样本会随机采样壳半径、壳厚度、噪声平滑尺度、碎片阈值和非对称强度。注意：该脚本的部分函数名和默认输出文件名仍保留 `Type_C` 字样；生成 Type A Shell 数据集时建议显式指定 `--dataset-output`。

常用生成命令示例：

```bash
python Type_A_SHELL_generating.py --generate-dataset --dataset-count 1000 --size 128 --seed 0 --dataset-output Type_A_SHELL_1000_128_GT.npy
```

主要参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--generate-dataset` | 启用批量数据集生成模式；不加时只预览单个壳状样本 | 关闭 |
| `--dataset-count` | 批量生成的样本数 | `1000` |
| `--dataset-output` | 输出 `.npy` 文件路径；建议显式指定 Type A Shell 文件名 | `Type_C_SHELL_1000_128_GT.npy` |
| `--radius` | 单图预览时的壳半径，单位为像素 | `10` |
| `--thickness` | 单图预览时的壳层厚度，单位为像素 | `2.0` |
| `--sigma` | 单图预览时随机噪声的平滑尺度；数值越大，碎片结构越平滑 | `2.0` |
| `--threshold` | 单图预览时的碎片门控阈值；数值越大，有效壳层碎片越少 | `0.15` |
| `--asymmetry` | 单图预览时的大尺度方位不对称强度 | `1.0` |
| `--size` | 输出图像边长像素数 | `128` |
| `--seed` | 随机种子，用于复现实验 | `None` |
| `--save` | 单图预览的保存路径 | 不保存 |

批量生成模式下，`radius` 会在 `3` 到 `10` 像素之间随机采样，`thickness` 会在 `0.2*radius` 到 `0.75*radius` 之间随机采样，`sigma` 在 `0` 到 `2` 之间随机采样，`threshold` 在 `0` 到 `0.2` 之间随机采样，`asymmetry` 在 `0` 到 `1` 之间随机采样。脚本运行结束后会打印保存文件名、数组形状、数据类型和像素最小/最大值。

### 4. `Type_A_DISK_generating.py`

`Type_A_DISK_generating.py` 用于生成 Type A 中的填充盘状本征数据集，模拟具有软边界、内部空洞、分形斑块和大尺度亮度梯度的扩展 disk morphology。脚本先生成圆形 top-hat 盘，再用高斯模糊软化边界；随后利用分形噪声阈值在盘内形成 cavity/fragmented 结构，并叠加一个方向性亮度梯度，最后归一化到采样得到的总流量。

批量生成时，输出为形状 `(样本数, 图像尺寸, 图像尺寸)` 的 `float32` 数组，并保存为 `.npy` 文件。每个样本会随机采样源中心、总流量、盘半径、边界模糊尺度、空洞阈值、空洞残余亮度、分形噪声谱指数、空洞对比度幂指数以及大尺度梯度强度。注意：该脚本中部分函数名、图标题和默认输出文件名仍保留 `Type_D` 字样；生成 Type A Disk 数据集时建议显式指定 `--dataset-output`。

常用生成命令示例：

```bash
python Type_A_DISK_generating.py --generate-dataset --dataset-count 1000 --intensity-min 20 --intensity-max 800 --radius-min 0.4 --radius-max 0.8 --blur-sigma-min 0.05 --blur-sigma-max 0.15 --seed 0 --dataset-output Type_A_DISK_1000_128_GT.npy
```

主要参数：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--generate-dataset` | 启用批量数据集生成模式；不加时只预览单个样本 | 关闭 |
| `--dataset-count` | 批量生成的样本数 | `1000` |
| `--dataset-output` | 输出 `.npy` 文件路径；建议显式指定 Type A Disk 文件名 | `Type_D_FragmentedDisk_1000_128_GT.npy` |
| `--random` | 单图预览时随机采样物理参数；批量生成本身始终随机采样 | 关闭 |
| `--center-mu` / `--center-sigma` | 源中心位置高斯分布的均值和标准差，单位为度 | `0.0` / `0.4` |
| `--max-offset` | 源中心相对视场中心的最大偏移，单位为度 | `2.4` |
| `--intensity` | 非随机单图预览时的总流量 | `200.0` |
| `--intensity-min` / `--intensity-max` | 批量生成或随机预览时总流量的对数均匀采样范围 | `20.0` / `800.0` |
| `--radius` | 非随机单图预览时的盘半径，单位为度 | `0.4` |
| `--radius-min` / `--radius-max` | 批量生成或随机预览时盘半径采样范围，单位为度 | `0.4` / `0.8` |
| `--blur-sigma` | 非随机单图预览时的边界高斯模糊尺度，单位为度 | `0.10` |
| `--blur-sigma-min` / `--blur-sigma-max` | 批量生成或随机预览时边界模糊尺度采样范围，单位为度 | `0.05` / `0.15` |
| `--cavity-threshold` | 非随机单图预览时的分形空洞阈值；越高则保留的高亮结构越少 | `0.45` |
| `--cavity-threshold-min` / `--cavity-threshold-max` | 批量生成或随机预览时空洞阈值采样范围 | `0.35` / `0.60` |
| `--cavity-floor` | 空洞区域的残余流量比例 | `0.05` |
| `--cavity-floor-min` / `--cavity-floor-max` | 批量生成或随机预览时空洞残余流量比例采样范围 | `0.0` / `0.12` |
| `--cavity-beta` | 分形空洞噪声的谱指数 | `2.0` |
| `--cavity-beta-min` / `--cavity-beta-max` | 批量生成或随机预览时分形噪声谱指数采样范围 | `1.6` / `2.6` |
| `--cavity-power` | 分形阈值后的结构对比度幂指数 | `1.0` |
| `--cavity-power-min` / `--cavity-power-max` | 批量生成或随机预览时结构对比度幂指数采样范围 | `0.8` / `1.6` |
| `--gradient-phi` | 非随机单图预览时的大尺度亮度梯度方向角，单位为弧度 | `pi/4` |
| `--gradient-strength-min` / `--gradient-strength-max` | 批量生成或随机预览时大尺度亮度梯度强度采样范围 | `0.15` / `0.55` |
| `--gradient-floor` | 梯度调制的最小乘性因子 | `0.25` |
| `--size` | 输出图像边长像素数 | `128` |
| `--fov` | 视场大小，单位为度 | `6.4` |
| `--seed` | 随机种子，用于复现实验 | `None` |
| `--save` | 单图预览的保存路径 | 不保存 |

脚本运行结束后会打印保存文件名、数组形状、数据类型、像素最小/最大值，以及每个样本总流量的最小值、最大值和均值。