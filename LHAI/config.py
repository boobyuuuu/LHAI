# This py file function: Store useful variables and configuration
# 要指定训练/推理的变量，有两种方式可以修改：1.在这个文件中修改；2.在 python train.py 后附加参数，例如 python train.py --MODEL_NAME=...

from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
load_dotenv()
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# ---- Train Parameters ----
EXP_NAME = "EXP_0_1"                                                                # 参数1：如果需要指定不同的实验，可以在这里修改
MODEL_NAME = "CNN"                                                                  # 参数2：如果需要指定不同的模型，可以在这里修改
MODEL_PATH = PROJ_ROOT / "LHAI" / "models" / f"{MODEL_NAME}_{EXP_NAME}.py"
DATA_DIR = PROJ_ROOT / "data" / "POISSON"                                           # 参数3：如果需要指定不同的数据集的文件名，可以在这里修改
DATA_NAME = "poisson_src_bkg.pkl.npy"                                               # 参数4：如果需要指定不同的数据集，可以在这里修改
DATA_PATH = DATA_DIR / DATA_NAME
SEED = 0                                                                            # 参数5：如果需要指定不同的随机种子，可以在这里修改
TRAINTYPE = "poissonsrc+bkg_highresorig"                                            # 参数6：如果需要指定不同的训练类型，可以在这里修改
FRAC_TRAIN = 0.8                                                                    # 参数7：如果需要指定不同的训练集比例，可以在这里修改
EPOCHS = 400                                                                        # 参数8：如果需要指定不同的训练轮数，可以在这里修改
BATCH_SIZE = 32                                                                     # 参数9：如果需要指定不同的批次大小，可以在这里修改
LATENTDIM = 64                                                                      # 参数10：如果需要指定不同的潜在维度，可以在这里修改(仅VAE模型)
LR_MAX = 5e-4                                                                       # 参数11：如果需要指定不同的学习率上限，可以在这里修改
LR_MIN = 5e-6                                                                       # 参数12：如果需要指定不同的学习率下限，可以在这里修改

# ---- Test Parameters ----
PRE_MODEL_PATH = PROJ_ROOT / "saves" / "MODEL"
PRE_DATA_PATH = PROJ_ROOT / "data" / "POISSON"
PRE_MODEL_NAME = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
RRE_MODEL = "CNN"
PRE_DATA_NAME = "poisson_src_bkg.pkl.npy"
PRE_SEED = 0
PRE_TRAINTYPE = "poissonsrc+bkg_highresorig"
PRE_FRAC_TRAIN = 0.8
PRE_BATCH_SIZE = 32
PRE_LATENT_DIM = 64

# ---- Eval Parameters ----
EVAL_EXP_NAME = "EXP_0_1"                                                                # 参数1：实验代号
EVAL_MODEL_NAME = "CNN"                                                                  # 参数2：模型在神经网络代码py文件中的class名称
EVAL_MODEL_PYPATH = PROJ_ROOT / "LHAI" / "models" / f"{EVAL_MODEL_NAME}_{EVAL_EXP_NAME}.py"
EVAL_MODEL_PTHNAME = "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"    # 参数3：需要评估的模型名称
EVAL_MODEL_PTHPATH = PROJ_ROOT / "saves" / "MODEL" / EVAL_MODEL_PTHNAME                  # 参数4：需要评估的模型保存的路径

EVAL_DATA_DIR = PROJ_ROOT / "data" / "POISSON"                                           # 参数3：如果需要指定不同的数据集的文件名
EVAL_DATA_NAME = "poisson_src_bkg.pkl.npy"                                               # 参数4：如果需要指定不同的数据集
EVAL_DATA_PATH = EVAL_DATA_DIR / EVAL_DATA_NAME
EVAL_SEED = 0                                                                            # 参数5：如果需要指定不同的随机种子

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass