from dataclasses import dataclass
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
    model_name: str = "CNN"
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    seed: int = 0
    frac: float = 0.98
    epochs: int = 10
    batch_size: int = 32
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0
    logpath: Path = ADDR_ROOT / "logs" / "train_cnn.log"

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.model_name}_{self.exp_name}.py"

    @property
    def data_path(self) -> Path:
        return self.data_dir / self.data_name

# ========== Predict Config ==========
@dataclass
class PredictConfig:
    model_name: str = "CNN"
    model_path: Path = ADDR_ROOT / "saves" / "MODEL"
    model_file: str = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
    data_dir: Path = ADDR_ROOT / "data" / "POISSON"
    data_name: str = "poisson_src_bkg.pkl.npy"
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
    model_name: str = "CNN"
    model_dir: Path = ADDR_ROOT / "codes" / "models"
    data_dir: Path = ADDR_ROOT / "data" / "Train"
    data_name: str = "xingwei_10000_64_train_v1.npy"
    model_weight_dir: Path = ADDR_ROOT / "saves" / "MODEL"
    model_weight_name: str = "CNN_EXP01_10epo_32bth_xingwei.pth"
    seed: int = 0
    frac: float = 0.98
    batch_size: int = 32
    epochs: int = 400
    lr_max: float = 5e-4
    lr_min: float = 5e-6
    datarange: float = 1.0

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.model_name}_{self.exp_name}.py"

    @property
    def data_path(self) -> Path:
        return self.data_dir / self.data_name

    @property
    def model_weight_path(self) -> Path:
        return self.model_weight_dir / self.model_weight_name


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
