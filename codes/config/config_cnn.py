from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

ADDR_CONFIG = Path(__file__).resolve().parents[0]
ADDR_ROOT = Path(__file__).resolve().parents[2]
logger.success(f"ADDR_CONFIG is: {ADDR_CONFIG}")
logger.info(f"ADDR_ROOT is: {ADDR_ROOT}")

# ========== Train Config ==========
@dataclass
class TrainConfig:
    exp_name: str = "EXP01"
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    model_name: str = "VDSR" # modified
    seed: int = 0
    frac: float = 0.98
    epochs: int = 400
    batch_size: int = 32
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0
    log_dir: Path = ADDR_ROOT / "saves" / "TRAIN" / "LOGS"
# ========== Predict Config ==========
@dataclass
class PredictConfig:
    model_name: str = "VDSR" # modified
    model_path: Path = ADDR_ROOT / "saves" / "MODEL"
    model_file: str = "VDSR_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth" # modified
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    seed: int = 0
    pred_type: str = "poissonsrc+bkg_highresorig"
    frac: float = 0.98
    batch_size: int = 32
    latent_dim: int = 64

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
    model_name: str = "VDSR" # modified
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    model_weight_dir: Path = ADDR_ROOT / "saves" / "MODEL"
    model_weight_name: str = "Last_VDSR_EXP01_400epo_32bth_xingwei.pth" # modified
    seed: int = 0
    frac: float = 0.98
    batch_size: int = 32
    epochs: int = 400
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0

# ========== Model Config ==========
@dataclass
class ModelConfig:
    model_params: dict = field(default_factory=lambda: {
        'CNN': {'jpt': 0},
        'CARN_v1': {
            'jpt': 0,
            'in_channels': 1,
            'out_channels': 1,
            'hid_channels': 64,
            'act_type': 'relu',
        },
        'CARN_v2': {'jpt': 0},
        'DRCN': {
            'in_channels': 1,
            'out_channels': 1,
            'recursions': 16,
            'hid_channels': 256,
            'act_type': 'relu',
            'arch_type': 'advanced',
            'use_recursive_supervision': False
        },
        'EDSR': {
            'in_channels': 1,
            'out_channels': 1,
            'num_blocks': 16,
            'num_feats': 64,
            'res_scale': 0.1,
            'act_type': 'relu'
        },
        'SRCNN_Transformer': {
            'jpt': None,               # 兼容占位
            'in_channels': 1,
            'out_channels': 1,
            'scale': 1,                 # 空间分辨率保持不变
            'shallow_feats': 64,        # 卷积特征通道数
            'patch_size': 4,            # Transformer patch 大小
            'transformer_layers': 2,    # Transformer 层数
            'transformer_heads': 4,     # Multi-head 注意力头数
            'transformer_embed': 64,    # Transformer embedding 维度
            'act_type': 'relu'          # 激活函数类型
        },
        'SwinIR_LHAI': {
            'in_ch': 1,
            'out_ch': 1,
            'img_size': 64,
            'embed_dim': 64,
            'depths': [2],          # RSTB 个数与每层深度
            'num_heads': [4],
            'window_size': 8,
            'mlp_ratio': 4.0
        },
        'ESPCN': {
            'jpt': 0,
            'in_channels': 1,
            'out_channels': 1,
            'n1': 64,
            'n2': 32,
            'act_type': 'prelu',
            'out_act_type': 'none'
        },
        'VDSR': {
            'jpt': 0, 
            'in_channels': 1,
            'out_channels': 1,
            'layer_num': 20,
            'hid_channels': 64
        },
        'LapSRN': {
            'jpt': 0,
            'in_channels': 1,
            'out_channels': 1,
            'hid_channels': 64,
            'fe_layers': 10
        }
    })
# 实例化默认配置
train_cfg = TrainConfig()
predict_cfg = PredictConfig()
eval_cfg = EvalConfig()

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
