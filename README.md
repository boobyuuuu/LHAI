# DATAGENERATING 响应数据生成说明

本目录用于把 `Type_A/Type_B/Type_C` 中的 GT 模板通过当前 LHAASO/KM2A 响应程序生成机器学习数据。输入 GT 支持：

```text
(N, 64, 64)
(N, 1, 64, 64)
```

如果输入是 `(N, 64, 64)`，脚本会自动转成响应程序需要的：

```text
(N, 1, 64, 64)
```

输出默认保存为：

```text
(N, 6, 64, 64)
```

通道含义：

```text
0: input / GT
1: excess map        # 仅卷积后的源响应
2: bkg map           # 仅背景（off）
3: bkg-on map        # 无泊松的源响应（excess + bkg）
4: poisson map       # 模拟观测图（on，源+背景叠加后做泊松抽样）
5: on-off map        # poisson_on - bkg，对应 LHAASO 数据处理中的 on-off 类型数据
```

## 运行环境

在服务器上先加载环境：

```bash
source /cvmfs/lhaaso.ihep.ac.cn/anysw/slc5_ia64_gcc73/external/envf.sh
cd /home/lhaaso/zhliu/LZH/response/Tools/NJU_AI/DATAGENERATING
```

## 推荐用法

示例：响应 `Type_A_COMPACT_1000_64_GT.npy` 的前 200 张图，F0 在 0.1 到 10.0 之间对数均匀采样，量级为 `1.e-16`，每批 100 张：

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_COMPACT_1000_64_GT.npy' \
  num 200 \
  fluxmin 0.1 \
  fluxmax 10.0 \
  fluxorder 16 \
  output './Type_A/Type_A_COMPACT_1000_64_conv.npy' \
  batchsize 100
```

这会生成：

```text
Type_A/Type_A_COMPACT_1000_64_conv.npy
Type_A/Type_A_COMPACT_1000_64_conv.json
Type_A/Type_A_COMPACT_1000_64_conv_work/
```

如果不想保留中间 batch 目录，可以默认不加 `keep_work`；如果需要排错或检查中间文件（如 ROOT 文件），建议加：

```bash
keep_work
```

或者使用 `--keep-work` 格式也可以。

## 参数说明

### 必填参数

```text
dataname <path>
```

输入 GT npy 文件路径。可以是绝对路径，也可以是相对于 `DATAGENERATING/` 的相对路径。

```text
num <int>
```

取输入文件的多少张图做响应。例如 `num 200` 表示处理 200 张图。

```text
start_index <int>
```

从输入文件的第几张图开始读取。默认为 `0`（从头开始）。

例如：
- `start_index 0 num 100`：处理 [0:100]
- `start_index 100 num 100`：处理 [100:200]
- `start_index 500 num 50`：处理 [500:550]

**用途：** 当需要对同一个输入文件的不同部分使用相同的流强序列时，可以多次调用脚本，每次指定不同的 `start_index`。参见 `run_TypeE.sh` 示例。

```text
output <path>
```

输出响应 npy 路径。脚本还会生成同名 `.json` 记录每张图的响应参数。

### 流强参数

```text
flux <float>
fluxmin <float>
fluxmax <float>
fluxorder <int>
fluxdist <uniform|log_uniform|const>
fluxshuffle
```

每张图的 F0 采样方式：

**分布类型（fluxdist）：**

- `uniform`：均匀分布，F0 在 [fluxmin, fluxmax] 之间均匀采样
- `log_uniform`：对数均匀分布，F0 在 [fluxmin, fluxmax] 之间对数均匀采样（默认）
- `const`：常数模式，所有 num 张图共享同一个 F0，值由 `flux` 参数指定（`fluxmin`/`fluxmax`/`fluxshuffle` 在该模式下被忽略）

**排序方式（fluxshuffle）：**

- 不加 `fluxshuffle`：F0 按升序排列，index 越大 F0 越大（默认）
- 加 `fluxshuffle`：F0 随机打乱，index 与 F0 大小无关
- `fluxdist=const` 时，所有值相同，`fluxshuffle` 无效

写入 `ParInit.yaml` 时格式为：

```yaml
F0: [F0_value, 0, upper, 0, 1.e-fluxorder]
```

例如：

```bash
fluxmin 0.1 fluxmax 10.0 fluxorder 16 fluxdist log_uniform
```

表示：

```yaml
F0: [0.1到10.0之间的对数均匀随机值（升序排列）, 0, 500, 0, 1.e-16]
```

对应 first LHAASO catalog 中常见的 `N0 = 0.1--10.0 × 10^-16 cm^-2 s^-1 TeV^-1` 范围。

或者使用 const 模式：

```bash
flux 2.0 fluxdist const fluxorder 16
```

表示所有 num 张图共享同一个 F0：

```yaml
F0: [2.0, 0, 500, 0, 1.e-16]
```

默认值：

```text
flux = None（仅当 fluxdist=const 时必填）
fluxmin = 0.1
fluxmax = 10.0
fluxorder = 16
fluxdist = log_uniform
fluxshuffle = False（不打乱，升序排列）
```

### 谱参数

```text
Epiv <float>
alpha <float>
```

默认：

```text
Epiv = 50
alpha = 3
```

写入：

```yaml
Epiv: 50
SEDModel:
  type: PL
  alpha: [3, 1.0, 5.0, 0]
```

### 探测器选择

```text
detector <km2a|wcda>
```

默认：

```text
detector = km2a
```

- `detector km2a`：生成 KM2A 响应，`Fit.yaml` 中 `KM2A.Active=1`、`WCDA.Active=0`。
- `detector wcda`：生成 WCDA 响应，`Fit.yaml` 中 `WCDA.Active=1`、`KM2A.Active=0`。

### KM2A 能量范围

```text
emin <float>
emax <float>
```

仅在 `detector km2a` 时使用，对应 `Fit.yaml` 中 KM2A 的：

```yaml
NbinUsed: [emin, emax]
```

单位是：

```text
log10(E/TeV)
```

默认：

```text
emin = 1.4
emax = 3.4
```

即大约：

```text
25 TeV -- 2512 TeV
```

### WCDA nhit 范围

```text
wcda_nhit_min <int>
wcda_nhit_max <int>
```

仅在 `detector wcda` 时使用，对应 `Fit.yaml` 中 WCDA 的：

```yaml
NbinUsed: [wcda_nhit_min, wcda_nhit_max]
```

默认：

```text
wcda_nhit_min = 1
wcda_nhit_max = 6
```

### ROI 参数

```text
ra_center <float|random|none>
dec_center <float|random|none>
```

ROI 固定为 6.4° × 6.4° 的矩形，对应 64×64、0.1°/pixel。

`Fit.yaml` 中写为：

```yaml
ROI:
  Include: [0, 1, RA_min, RA_max, Dec_min, Dec_max]
```

含义：

```text
Include[0] = 0: 赤道坐标
Include[1] = 1: 矩形 ROI
RA_min / RA_max: 赤经范围
Dec_min / Dec_max: 赤纬范围
```

默认：

```text
ra_center = random   # 0 到 360 度均匀随机
dec_center = 22      # 固定 22 度
```

如果想固定赤经：

```bash
ra_center 75.2
```

如果想随机赤纬：

```bash
dec_center random
```

注意：当前响应程序一次 `Src_Convo_Template` 运行只能对应一个 `Fit.yaml`，也就是一个 ROI。脚本按 batch 运行，所以：

- `batchsize=100` 时，每 100 张图共享同一个 ROI 中心。
- 如果想每张图都有独立随机 ROI，请设置：

```bash
batchsize 1
```

### batch 参数

```text
batchsize <int>
```

每次调用响应程序处理多少张图。默认：

```text
batchsize = 100
```

较大的 batch 更快，但中间 ROOT 文件更大；较小的 batch 更灵活，尤其适合随机 ROI。

### 随机种子

```text
seed <int>
batch_time_seed
```

默认使用固定 seed，可复现。固定 seed 模式控制：

- F0 随机采样；
- 随机 RA/Dec 中心；
- 泊松随机 seed。

如果传入 `seed 0`，这些随机操作都会使用 seed `0`。例如：

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_COMPACT_1000_64_GT.npy' \
  num 200 \
  seed 0 \
  output './Type_A/Type_A_COMPACT_200_seed0_conv.npy' \
  batchsize 100
```

如果希望每个 batch 使用不同随机种子，可以加 `batch_time_seed`。脚本会在每个 batch 开始时取当前时间作为该 batch 的 seed，并且该 batch 内的 F0 随机采样、随机 ROI、Poisson seed 都使用同一个时间 seed：

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_COMPACT_1000_64_GT.npy' \
  num 200 \
  batch_time_seed \
  output './Type_A/Type_A_COMPACT_200_timeseed_conv.npy' \
  batchsize 100
```

默认：

```text
seed = 20260514
batch_time_seed = False
```

### workdir 参数

```text
workdir <path>
```

写入 `Fit.yaml` 的 `WorkDir`。默认：

```text
/home/lhaaso/zhliu/LZH/response
```

## 输出 JSON

每次运行会生成一个和输出 npy 同名的 JSON，例如：

```text
Type_A/Type_A_COMPACT_1000_64_conv.json
```

里面记录：

- 输入/输出文件；
- num、batchsize；
- 能量参数；
- flux 分布；
- ROI 设置；
- 每个 sample 的：
  - F0；
  - flux order；
  - Epiv；
  - alpha；
  - RA center；
  - Dec center；
  - batch id。

## 更多例子

### 只测试前 2 张图

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_COMPACT_1000_64_GT.npy' \
  num 2 \
  output './Type_A/test_compact_2_conv.npy' \
  batchsize 2 \
  --keep-work
```

### 固定 ROI 中心

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_GAUSSIAN_1000_64_GT.npy' \
  num 200 \
  fluxmin 0.1 fluxmax 10.0 fluxorder 16 \
  ra_center 75.2 dec_center 25.0 \
  output './Type_A/Type_A_GAUSSIAN_200_64_conv.npy' \
  batchsize 100
```

### 每张图随机天区

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_SHELL_1000_64_GT.npy' \
  num 50 \
  ra_center random dec_center random \
  output './Type_A/Type_A_SHELL_50_randomROI_conv.npy' \
  batchsize 1
```

### 修改能量范围

```bash
bash run_response.sh \
  dataname './Type_A/Type_A_COMPACT_1000_64_GT.npy' \
  num 200 \
  emin 1.8 emax 3.4 \
  output './Type_A/Type_A_COMPACT_200_highE_conv.npy' \
  batchsize 100
```

### 对数均匀分布，随机打乱，保留中间文件

```bash
bash run_response.sh \
dataname './Type_A/Type_A_COMPACT_1000_64_GT.npy' \
num 200 \
fluxmin 0.1 \
fluxmax 10.0 \
fluxorder 16 \
fluxdist log_uniform \
fluxshuffle \
keep_work \
output './Type_A/Type_A_COMPACT_200_loguniform_shuffle.npy' \
batchsize 100
```

### 所有图共享同一个 flux（const 模式）

例如对 `Exp2_TwoPointSource/EXP2SEP_1000_64_GT.npy` 的 1000 张图都使用 `F0 = 2.0 × 10^-16`，DEC 固定 22°、RA 随机：

```bash
bash run_response.sh \
  dataname './Exp2_TwoPointSource/EXP2SEP_1000_64_GT.npy' \
  num 1000 \
  flux 2.0 \
  fluxdist const \
  fluxorder 16 \
  dec_center 22 \
  output './Exp2_TwoPointSource/EXP2SEP_1000_64_conv.npy' \
  batchsize 100
```

`fluxdist=const` 时只读取 `flux` 参数，`fluxmin`/`fluxmax`/`fluxshuffle` 会被忽略。