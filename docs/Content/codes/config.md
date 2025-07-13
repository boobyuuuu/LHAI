# LHAI 函数模块 — 配置参数 Config 部分

本模块统一定义并管理训练（Train）、预测（Predict）与评估（Eval）所需的全部参数，以便于结构化调用、快速调整与可维护性。

---

## 配置体系核心结构

- 基于 `@dataclass` 构建：每一类配置（训练、预测、评估）独立封装为一个数据类，参数定义简洁、清晰。
- 路径统一由 `Path` 类型管理，便于跨平台兼容与路径拼接安全性。
- 通过 `dotenv` + `Path(__file__)` 实现对项目路径的自动解析，无需手动配置根目录。

---

## 文件路径自动处理

```python
ADDR_CONFIG = Path(__file__).resolve().parents[0]
ADDR_ROOT = Path(__file__).resolve().parents[2]
```

* `ADDR_CONFIG`: 当前配置文件所在的 `config` 目录路径。
* `ADDR_ROOT`: 项目的根目录，用于构造相对路径，避免路径硬编码。

---

## 1. 训练配置：`TrainConfig`

用于控制模型训练阶段的所有核心超参数与路径设置。

```python
@dataclass
class TrainConfig:
    exp_name: str = "EXP01"  # 实验名称，用于日志记录与模型命名
    model_name: str = "CNN"  # 模型名称，控制模型构建与保存文件前缀
    model_dir: Path = ADDR_ROOT / "codes" / "models"  # 模型定义代码目录
    data_dir: Path = ADDR_ROOT / "data" / "Train"     # 训练数据存放目录
    data_name: str = "xingwei_10000_64_train_v1_processed.npy"  # 数据文件名
    seed: int = 0             # 随机种子，确保可复现性
    frac: float = 0.98        # 训练集划分比例（剩余部分用于测试集）
    epochs: int = 10          # 训练轮数
    batch_size: int = 32      # 批次大小
    lr_max: float = 5e-4      # 初始最大学习率
    lr_min: float = 5e-6      # 最小学习率（用于调度）
    datarange: float = 1.0    # 数据归一化范围（1.0表示[0,1]，2.0表示[-1,1]）
    logpath: Path = ADDR_ROOT / "logs" / "train_cnn.log"  # 日志保存路径
```

---

## 2. 预测配置：`PredictConfig`

用于模型推理或部署时的配置参数。

```python
@dataclass
class PredictConfig:
    model_name: str = "CNN"  # 模型名称
    model_path: Path = ADDR_ROOT / "saves" / "MODEL"  # 模型权重存放路径
    model_file: str = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
    data_dir: Path = ADDR_ROOT / "data" / "POISSON"  # 待预测数据目录
    data_name: str = "poisson_src_bkg.pkl.npy"  # 待预测数据文件
    seed: int = 0
    pred_type: str = "poissonsrc+bkg_highresorig"  # 用于标记推理结果来源
    frac: float = 0.98
    batch_size: int = 32
    latent_dim: int = 64  # 若使用 VAE/扩散模型等，需指定潜在向量维度
```

> 该类还提供两个便捷属性方法：

```python
@property
def data_path(self) -> Path:
    return self.data_dir / self.data_name

@property
def full_model_path(self) -> Path:
    return self.model_path / self.model_file
```

* 自动拼接模型路径与数据路径，调用更简洁，避免冗余代码。

---

## 3. 评估配置：`EvalConfig`

用于模型评估流程，如加载指定模型权重、在验证集上计算各类指标。

```python
@dataclass
class EvalConfig:
    exp_name: str = "EXP01"
    model_name: str = "CNN"
    model_dir: Path = ADDR_ROOT / "codes" / "models"  # 模型定义目录
    data_dir: Path = ADDR_ROOT / "data" / "Train"     # 测试数据目录
    data_name: str = "xingwei_10000_64_train_v1.npy"  # 测试数据文件名
    model_weight_dir: Path = ADDR_ROOT / "saves" / "MODEL"  # 权重目录
    model_weight_name: str = "CNN_EXP01_10epo_32bth_xingwei.pth"  # 指定模型文件
    seed: int = 0
    frac: float = 0.98
    batch_size: int = 32
    epochs: int = 400
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0
```

---

## 日志打印支持

模块末尾提供命令行运行支持：

```python
if __name__ == "__main__":
```

* 若直接执行 `config_cnn.py`，将使用 `loguru` 配合 `tqdm` 实时格式化打印全部参数。
* 所有字段均通过反射方式循环打印，支持扩展。

输出示例：

```shell
--- Train Config ---
exp_name = EXP01
model_name = CNN
...
logpath = /.../logs/train_cnn.log
```
