from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

ADDR_CONFIG = Path(__file__).resolve().parents[0]
ADDR_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    logger.success(f"ADDR_CONFIG is: {ADDR_CONFIG}")
    logger.info(f"ADDR_ROOT is: {ADDR_ROOT}")

# ========== Train Config ==========
@dataclass
class TrainConfig:
    exp_name: str = "EXP01"
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    model_name_diffusion: str = "DDPM"
    model_name_unet: str = "UNET"
    seed: int = 0
    frac: float = 0.80
    epochs: int = 2
    batch_size: int = 32
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0
# ========== Predict Config ==========
@dataclass
class PredictConfig:
    exp_name: str = "EXP01"
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    model_name_diffusion: str = "DDPM"
    model_name_unet: str = "UNET"
    model_weight_name: str = "Last_DDPM_EXP01_400epo_32bth_xingwei.pth"
    data_dir: Path = ADDR_ROOT / "data" / "Evaluation"
    data_name: str = "nhit100_1_64_val.npy"
    seed: int = 0
    epochs: int = 400
    batch_size: int = 1
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0
    LoB: str = "Last" # "Best" or "Last"
    dataname: str = "xingwei"

    @property
    def data_path(self) -> Path:
        return self.data_dir / self.data_name

    @property
    def full_model_path(self) -> Path:
        return self.model_path / self.model_file

# ========== Eval Config ==========
@dataclass
class EvalConfig:
    exp_name: str = "EXP01"
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    model_name_diffusion: str = "DDPM"
    model_name_unet: str = "UNET"
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    seed: int = 0
    frac: float = 0.80
    epochs: int = 2
    batch_size: int = 32
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0
    LoB: str = "Last" # "Best" or "Last"
    dataname: str = "xingwei"
    
@dataclass
class ModelConfig:
    model_params: dict = field(default_factory=lambda: {
        'UNET': {
            'jpt': None,                    # 兼容占位
            'in_channels': 2,               # 条件扩散输入：LR 与 x_t 拼接
            'channels': [32, 64, 128],
            'base_channels': [256, 256],
            'channel_attention': [False, False, False],
            'out_channels': 1,              # 预测噪声
            'position_embedding_dim': 256,  # 与 diffusion 的 pos_emb_dim 对齐
        },
        'UNET_v2': {
            'jpt': None,                    # 兼容占位
            'in_channels': 2,               # 条件扩散输入：LR 与 x_t 拼接
            'channels': [32, 64, 128],
            'base_channels': [256, 256],
            'channel_attention': [False, False, False],
            'out_channels': 1,              # 预测噪声
            'position_embedding_dim': 256,  # 与 diffusion 的 pos_emb_dim 对齐
        },
        'DDPM': {
            'noise_steps': 2000,            # 扩散过程步数
            'beta_start': 1e-6,             # 噪声调度起始值
            'beta_end': 0.01,               # 噪声调度结束值
            'img_size': 64,                 # 图像分辨率 (H=W=64)
            'device': None,                 # 默认为 None，会在类内自动选择 CUDA/CPU
            'pos_emb_dim': 256,             # 若需要位置编码，设置 embedding 维度
            'conditional': True              # 是否启用条件扩散（拼接 LR 输入）
        },
        'DDPM_v2': {
            'noise_steps': 2000,            # 扩散过程步数
            'beta_start': 1e-6,             # 噪声调度起始值
            'beta_end': 0.01,               # 噪声调度结束值
            'img_size': 64,                 # 图像分辨率 (H=W=64)
            'device': None,                 # 默认为 None，会在类内自动选择 CUDA/CPU
            'pos_emb_dim': 256,             # 若需要位置编码，设置 embedding 维度
            'conditional': True              # 是否启用条件扩散（拼接 LR 输入）
        },
        'DDPM_Transformer': {
            'noise_steps': 2000,        # 扩散过程的步数 (T)
            'beta_start': 1e-6,         # 噪声调度起始值
            'beta_end': 0.01,           # 噪声调度结束值
            'img_size': 64,             # 图像输入/输出分辨率 (H=W=64)
            'device': None,             # 默认 None，类内部会自动选择 "cuda" 或 "cpu"
            'pos_emb_dim': 256,         # 时间步位置编码维度
            'conditional': True         # 是否启用条件扩散 (输入=LR+噪声)
        }
    })

# 实例化默认配置
train_cfg = TrainConfig()
predict_cfg = PredictConfig()
eval_cfg = EvalConfig()
model_cfg = ModelConfig()

if __name__ == "__main__":
    try:
        from tqdm import tqdm
        logger.remove(0)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:
        pass

    logger.info("========== 当前配置参数 ==========")

    for cfg_name, cfg_obj in zip(["Train", "Predict", "Eval"], [train_cfg, predict_cfg, eval_cfg]):
        logger.info(f"--- {cfg_name} Config ---")
        for field in cfg_obj.__dataclass_fields__:
            logger.info(f"{field} = {getattr(cfg_obj, field)}")
        # 打印 property 值
        for attr in dir(cfg_obj):
            if not attr.startswith('_') and not attr in cfg_obj.__dataclass_fields__:
                try:
                    value = getattr(cfg_obj, attr)
                    if isinstance(value, (Path, str, float, int)):
                        logger.info(f"{attr} = {value}")
                except:
                    pass

    logger.success("========== 配置参数输出完毕 ==========")
