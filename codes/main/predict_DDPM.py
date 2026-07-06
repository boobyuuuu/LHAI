# 所有DIFFUSION方案模型通用测试框架

# ---- 01 Improt Libraries ----
# ---- 1-1 Libraries for Path and Logging ----
import os
import sys
import typer
from tqdm import tqdm
from pathlib import Path
from loguru import logger
ADDR_ROOT = Path(__file__).resolve().parents[2]
logger.success(f"ADDR_ROOT path is: {ADDR_ROOT}")
ADDR_CODE = Path(__file__).resolve().parents[1]
sys.path.append(str(ADDR_ROOT))
logger.success(f"ADDR_CODE path is: {ADDR_CODE}")
# ---- 1-2 Libraries for Configuration and Modules ----
from codes.function.Log import log
import codes.function.Train as Train
import codes.function.Loss as lossfunction
from codes.config.config_DDPM import PredictConfig
from codes.config.config_DDPM import ModelConfig
from codes.function.Dataset import ImageDataset, DataModule, SingleImageDataset
# ---- 1-3 PyTorch ----
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split, ConcatDataset
# ---- 1-4 Others ----
import scipy
import importlib
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

import deeplay as dl
# ---- 1-5 eval ----
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE

# ---- 02 Define the main function ----
prid_cfg = PredictConfig()
model_cfg = ModelConfig()
app = typer.Typer()
@app.command()
def main(
    exp_name: str = prid_cfg.exp_name,                      # para01：实验名称 default: "EXP01"
    model_dir: Path = prid_cfg.model_dir,               # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_name_diffusion: str = prid_cfg.model_name_diffusion,              # para02：模型名称 default: "DIFFUSION"
    model_name_unet: str = prid_cfg.model_name_unet,                            # para02：模型名称 default: "UNET"
    model_weight_name: str = prid_cfg.model_weight_name,        # para04：模型权重文件名 default: "Last_DDPM_EXP01_400epo_32bth_xingwei.pth"
    data_dir: Path = prid_cfg.data_dir,                 # para05：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = prid_cfg.data_name,                # para06：数据文件名 default: "xingwei_10000_64_train_v1.npy"
    seed: int = prid_cfg.seed,                              # para11：随机种子 default: 0
    epochs: int = prid_cfg.epochs,                          # para14：训练轮数（如需评估多个 epoch） default: 400
    batch_size: int = prid_cfg.batch_size,                  # para13：批次大小 default: 32
    lr_max: float = prid_cfg.lr_max,                        # para15：最大学习率 default: 5e-4
    lr_min: float = prid_cfg.lr_min,                        # para16：最小学习率 default: 5e-6
    datarange: float = prid_cfg.datarange,
    LoB: str = prid_cfg.LoB,                                # para17：选择加载Best还是Last模型 default: "Last"
    dataname: str = prid_cfg.dataname                       # para18：数据集名称缩写 default: "xingwei"
):
    # ==== 2-1 Initialization  ====
    # train 自定义参数
    data_path = data_dir / data_name
    model_path = model_dir / f"{model_name_diffusion}.py"
    model_weight_name = model_weight_name
    model_weight_dir = ADDR_ROOT / "saves" / "MODEL" / model_name_diffusion
    model_weight_path = model_weight_dir / model_weight_name

    # exp 实验参数
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.success("========= 2-1 参数加载完成 =========")
    
    # ==== 2-2 Data: trainloader & testloader ====
    dataset = SingleImageDataset(data_path, data_range=datarange)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in testloader:
        # batch shape: [B, 1, 64, 64]
        img_tensor = batch  # 直接取出
        img_shape = tuple(img_tensor.shape)

        img_numpy = img_tensor.numpy()
        img_min = float(img_numpy.min())
        img_max = float(img_numpy.max())
        img_sample = img_numpy  # 或者 img_numpy[0] 如果只打印第一张
        break

    logger.info(f"""
    ====================== 数据参数 ======================
    Output of data from Batch 1

    - img shape     : {img_shape} [批次, 通道数, 高度, 宽度]
    - datarange        : 最小值 = {img_min:.6f}, 最大值 = {img_max:.6f}
    - image output :

    {np.array2string(img_sample, precision=4, suppress_small=True, threshold=64)}
    ===============================================================
    """)

    logger.success("========= 2-2 数据加载完成 =========")

    # ==== 2-3 Initialize the model ====
    # DIFFUSION
    model_params = model_cfg.model_params
    params_diffusion = model_params[model_name_diffusion]
    params_unet = model_params[model_name_unet]

    sys.path.append(str(model_dir))
    module_diffusion = importlib.import_module(model_name_diffusion)
    DIFFUSION = getattr(module_diffusion, model_name_diffusion)

    diffusion = DIFFUSION(**params_diffusion)

    # Unet
    model_params = model_cfg.model_params
    params_unet = model_params[model_name_unet]
    sys.path.append(str(model_dir))
    module_unet = importlib.import_module(model_name_unet)
    UNET = getattr(module_unet, model_name_unet)

    unet = UNET(**params_unet).to(device)
    unet.build()
    unet.to(device)
    
    # 加载权重
    unet.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    # loss
    trainingloss = lossfunction.msejsloss

    logger.success(f"========= 2-3 模型、模型参数与loss加载完成 =========")
    
    # ==== 2-4 PREDICT ====

    # save path
    dataname = data_name.split("_")[0]
    save_dir_pred = ADDR_ROOT / "saves" / "PREDICT" / model_name_diffusion
    if not os.path.exists(save_dir_pred):
        os.makedirs(save_dir_pred)

    # logger output
    format_model_params = Train.format_model_params
    torch.set_printoptions(precision=10)
    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    loss_name = trainingloss.__name__
    model_params_str_diffusion = format_model_params(model_params[model_name_diffusion])
    model_params_str_unet = format_model_params(model_params[model_name_unet])
    filetmp = np.load(data_path, allow_pickle=True)
    filelen = int(filetmp.shape[0])
    del filetmp
    train_msg = f"""
    ====================== 评估参数 ======================
    🔧 配置信息概览：
    - traintime               : {train_time}
    - exp_name                : {exp_name}
    - model_name              : {model_name_diffusion} + {model_name_unet}
    - model_weight_name       : {model_weight_path}
    - data_name               : {data_name}（{dataname}）
    - model_path              : {model_path}
    - data_path               : {data_path}
    - seed                    : {seed}
    - datalength              : {filelen}
    - epochs                  : {epochs}
    - batch_size              : {batch_size}
    - datarange               : {datarange}
    - learnrate               : 最小 = {lr_min:.1e}, 最大 = {lr_max:.1e}
    - lossname                : {loss_name}
    - device                  : {device}({gpu_name})
    - model_params_diffusion  : 
    
    {model_params_str_diffusion}
    - model_params_unet       :

    {model_params_str_unet}
    ==============================================================
    """
    logger.info(train_msg)


    # 推理（生成高分辨图像序列）
    for batch in testloader:
        input_img = batch.to(device)  # shape: [1, 1, 64, 64]
        break

    with torch.no_grad():
        generated_images = diffusion.reverse_diffusion(
            model=unet,
            n_images=1,
            n_channels=1,
            input_image=input_img,
            save_time_steps=None
        )
        
    print(generated_images.shape)
    
    # ---------------------------
    # 2. Convert Tensor → numpy
    # ---------------------------
    input_np = input_img[0,0].detach().cpu().numpy()
    output_np = generated_images[0,0].detach().cpu().numpy()

    # ---------------------------
    # 3. Save as .npy
    # ---------------------------
    # np.save(save_dir_pred / f"{dataname}_input.npy", input_np.astype(np.float32))
    # np.save(save_dir_pred / f"{dataname}_SR.npy", output_np.astype(np.float32))

    # ---------------------------
    # 4. Save as Images (.png)
    # ---------------------------
    plt.figure(figsize=(4,4))
    plt.imshow(input_np, cmap="viridis")
    plt.axis("off")
    plt.title("Input Image")
    plt.savefig(save_dir_pred / f"{dataname}_input.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4,4))
    plt.imshow(output_np, cmap="viridis")
    plt.axis("off")
    plt.title("Super-Resolved Output")
    plt.savefig(save_dir_pred / f"{dataname}_SR.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"[Saved] Results stored in: {save_dir_pred}")

if __name__ == "__main__":
    app()
