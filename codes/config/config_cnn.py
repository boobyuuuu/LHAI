# cnn模型配置文件
# 要指定训练/推理的变量，有两种方式可以修改：1.在这个文件中修改；2.在 python train.py 后附加参数，例如 python train.py --MODEL_NAME=...

from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
load_dotenv()
ADDR_CONFIG = Path(__file__).resolve().parents[0]
ADDR_ROOT = Path(__file__).resolve().parents[2]
logger.success(f"ADDR_CONFIG is: {ADDR_CONFIG}")
logger.info(f"ADDR_ROOT is: {ADDR_ROOT}")

# ==== Train Parameters ====
TRAIN_EXP_NAME        = "EXP01"
TRAIN_MODEL_NAME      = "CNN"
TRAIN_MODEL_PY        = ADDR_ROOT / "codes" / "models" / f"{TRAIN_MODEL_NAME}_{TRAIN_EXP_NAME}.py"
TRAIN_DATA_DIR        = ADDR_ROOT / "data" / "Train"
TRAIN_DATA_NAME       = "xingwei_10000_64_train_v1.npy"
TRAIN_DATA_PATH       = TRAIN_DATA_DIR / TRAIN_DATA_NAME
TRAIN_SEED            = 0
TRAIN_FRAC            = 0.8
TRAIN_EPOCHS          = 400
TRAIN_BATCH_SIZE      = 32
TRAIN_LR_MAX          = 5e-4
TRAIN_LR_MIN          = 5e-6

# ==== Test Parameters ====
PRED_MODEL_NAME       = TRAIN_MODEL_NAME
PRED_MODEL_PATH       = ADDR_ROOT / "saves" / "MODEL"
PRED_MODEL_FILE       = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
PRED_DATA_DIR         = ADDR_ROOT / "data" / "POISSON"
PRED_DATA_NAME        = "poisson_src_bkg.pkl.npy"
PRED_DATA_PATH        = PRED_DATA_DIR / PRED_DATA_NAME
PRED_SEED             = 0
PRED_TYPE             = "poissonsrc+bkg_highresorig"
PRED_FRAC             = 0.8
PRED_BATCH_SIZE       = 32
PRED_LATENT_DIM       = 64

# ==== Eval Parameters ====
EVAL_EXP_NAME         = TRAIN_EXP_NAME
EVAL_MODEL_NAME       = TRAIN_MODEL_NAME
EVAL_MODEL_PY         = ADDR_ROOT / "codes" / "models" / f"{EVAL_MODEL_NAME}_{EVAL_EXP_NAME}.py"
EVAL_MODEL_FILE       = PRED_MODEL_FILE
EVAL_MODEL_PATH       = ADDR_ROOT / "saves" / "MODEL" / EVAL_MODEL_FILE
EVAL_DATA_DIR         = ADDR_ROOT / "data" / "POISSON"
EVAL_DATA_NAME        = "poisson_src_bkg.pkl.npy"
EVAL_DATA_PATH        = EVAL_DATA_DIR / EVAL_DATA_NAME
EVAL_SEED             = 0


if __name__ == "__main__":
    try:
        from tqdm import tqdm
        logger.remove(0)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:
        pass

    logger.info("========== 当前配置参数 ==========")

    config_items = dict(globals())

    for var_name, var_value in config_items.items():
        if var_name.isupper():
            logger.info(f"{var_name} = {var_value}")

    logger.success("========== 配置参数输出完毕 ==========")
