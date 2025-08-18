# 所有DIFFUSION方案模型通用训练框架

# ---- 01 Improt Libraries ----
# ---- 1-1 Libraries for Path and Logging ----
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import sys
ADDR_ROOT = Path(__file__).resolve().parents[2]
logger.success(f"ADDR_ROOT path is: {ADDR_ROOT}")
ADDR_CODE = Path(__file__).resolve().parents[1]
sys.path.append(str(ADDR_ROOT))
logger.success(f"ADDR_CODE path is: {ADDR_CODE}")
# ---- 1-2 Libraries for Configuration and Modules ----
from codes.config.config_diffusion import TrainConfig
from codes.function.Dataset import ImageDataset
import codes.function.Loss as lossfunction
from codes.function.Log import log
import codes.function.Train as Train
from codes.models.DIFFUSION import EnhancedUNetWrapper
from codes.models.DIFFUSION import Diffusion
from codes.models.DIFFUSION import positional_encoding
from codes.models.DIFFUSION import prepare_data
# ---- 1-3 Libraries for pytorch and others ----
import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np
import deeplay as dl
import time
from datetime import timedelta
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE
import seaborn as sns

# ---- 02 Define the main function ----
train_cfg = TrainConfig()
app = typer.Typer()
@app.command()
def main(
    exp_name: str = train_cfg.exp_name,                  # para01：实验名称 default: "EXP01"
    data_dir: Path = train_cfg.data_dir,                 # para05：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = train_cfg.data_name,                # para06：数据文件名 default: "xingwei_10000_64_train_v1.npy"
    model_dir: Path = train_cfg.model_dir,               # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_name: str = train_cfg.model_name,              # para02：模型名称 default: "DIFFUSION"
    seed: int = train_cfg.seed,                          # para08：随机种子 default: 0
    frac: float = train_cfg.frac,                        # para09：训练集比例 default: 0.8
    epochs: int = train_cfg.epochs,                      # para10：训练轮数 default: 400
    batch_size: int = train_cfg.batch_size,              # para11：批次大小 default: 32
    lr_max: float = train_cfg.lr_max,                    # para12：最大学习率 default: 5e-4
    lr_min: float = train_cfg.lr_min,                    # para13：最小学习率 default: 5e-6
    datarange: float = train_cfg.datarange,              # para14：数据范围 default: 1.0
    position_encoding_dim: int = train_cfg.position_encoding_dim,   # para15：位置编码维度 default: 256
    noise_steps: int = train_cfg.noise_steps,            # para16：噪声步数 default: 2000
    EVALUATE_METRICS: bool = train_cfg.EVALUATE_METRICS, # para17：是否评估指标 default: False
    log_dir: Path = train_cfg.log_dir,                   # para18：日志文件夹 default: ADDR_ROOT / "logs"
):
    #【重要】根据命令行输入重新定义参数
    data_path = data_dir / data_name
    model_path = model_dir / f"{model_name}.py"
    logpath = log_dir / f"trainlog_{model_name}"
    # ---- 2-1 Load the parameter ----
    logger.info("========== 当前训练参数 ==========")
    for idx, (key, value) in enumerate(locals().items(), start=1):
        logger.info(f"{idx:02d}. {key:<20}: {value}")
    torch.manual_seed(seed)
    # spec = importlib.util.spec_from_file_location("module.name", model_path)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules["module.name"] = module
    # spec.loader.exec_module(module)
    # MODEL = getattr(module, model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_sim = data_name.split("_")[0]
    model_save_name = f"{model_name}_{exp_name}_{epochs}epo_{batch_size}bth_{data_sim}"
    logger.success("✅ 参数加载完成（Step 2-1）")
    
    # ---- 2-2 Load data ----
    filetmp = np.load(data_path, allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    num_to_learn = int(filelen)

    dataset = ImageDataset(num_to_learn, data_path, inverse=False,data_range=datarange)
    trainset, testset = random_split(
        dataset,
        lengths=[int(frac * len(dataset)), len(dataset) - int(frac * len(dataset))],
        generator=torch.Generator().manual_seed(0)
    )

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for batch_idx, (blurry_img, original_img) in enumerate(trainloader):
        if batch_idx == 0:
            blurry_img_shape = blurry_img.shape  # 示例：(32, 1, 64, 64)
            original_img_shape = original_img.shape
            blurry_img_numpy = blurry_img[1].squeeze().detach().cpu().numpy()
            blurry_img_min = blurry_img_numpy.min()
            blurry_img_max = blurry_img_numpy.max()
            blurry_img_sample = blurry_img_numpy
            break

    logger.info(f"""
    ====================== 📊 数据集统计信息 ======================
    样本来源：第一个训练 mini-batch（Batch 1）

    - 🖼️ 模糊图像张量尺寸     : {blurry_img_shape}（格式：[批次, 通道数, 高度, 宽度]）
    - 🖼️ 原始图像张量尺寸     : {original_img_shape}（格式：[批次, 通道数, 高度, 宽度]）
    - 🔢 模糊图像像素取值范围 : 最小值 = {blurry_img_min:.6f}, 最大值 = {blurry_img_max:.6f}
    - 🧪 模糊图像样本（索引 1）二维数据如下（截取）：
    {np.array2string(blurry_img_sample, precision=4, suppress_small=True, threshold=64)}
    ===============================================================
    """)

    logger.success("✅ 数据加载完成（Step 2-2）")

    # ---- 2-3 Initialize the model, loss function and optimizer ----
    # 模型
    unet = dl.AttentionUNet(
        in_channels=2,
        channels=[32, 64, 128],
        base_channels=[256, 256],
        channel_attention=[False, False, False],
        out_channels=1,
        position_embedding_dim=position_encoding_dim,
    )
    unet.build()
    unet.to(device)
    diffusion = Diffusion(
        noise_steps=noise_steps,
        img_size=64,
        beta_start=1e-6,
        beta_end=0.01,
    )
    # 优化器
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr_max)
    lr_lambda = lambda epoch: lr_min / lr_max + 0.5 * (1 - lr_min / lr_max) * (1 + np.cos(np.pi * epoch / epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # 损失函数
    criterion = lossfunction.msejsloss

    logger.success("✅ 模型、损失函数、优化器加载完成（Step 2-3）")

    # ---- 2-4 Start Training ----
    train = Train.train_diffusion  # 确认train_diffusion函数签名与之前定义的train一致

    ms_ssim_metric = MS_SSIM(
        data_range=datarange, kernel_size=7, betas=(0.0448, 0.2856, 0.3001)
    ).to(device)
    ssim_metric = SSIM(data_range=datarange).to(device)
    psnr_metric = PSNR(data_range=datarange).to(device)
    mae_metric = MAE().to(device)

    train_loss = []
    mae_results = []
    ms_ssim_results = []
    ssim_results = []
    psnr_results = []
    nrmse_results = []

    torch.set_printoptions(precision=10)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    optimizer_name = optimizer.__class__.__name__                      # 优化器类名，例如 AdamW
    loss_name = criterion.__name__                                  # 损失函数名称，例如 msejsloss

    train_msg = f"""
    ====================== 🚀 开始训练 ======================
    🔧 配置信息总览：
    📦 实验名称             : {exp_name}
    🧠 模型名称             : {model_name}
    📁 模型脚本路径         : {model_path}
    📂 数据文件路径         : {data_path}
    📊 数据集切分比例       : 训练集 {frac*100:.1f}% / 测试集 {100-frac*100:.1f}%
    📈 样本总数             : {filelen}
    🔁 总训练轮数 (Epochs)  : {epochs}
    📦 批次大小 (BatchSize)  : {batch_size}
    🌱 随机种子 (Seed)      : {seed}
    🔢 数据归一化范围       : {datarange}
    🧩 位置编码维度         : {position_encoding_dim}
    🎲 噪声步数             : {noise_steps}
    🔍 是否评估指标         : {EVALUATE_METRICS}
    📉 学习率策略 (Cosine)  : 最小 = {lr_min:.1e}, 最大 = {lr_max:.1e}
    🧪 损失函数 (Loss)      : {loss_name}
    🛠️ 优化器 (Optimizer)  : {optimizer_name}
    💻 使用设备 (Device)    : {device} ({gpu_name})
    ==============================================================
    """

    logger.info(train_msg)

    train(
        unet=unet,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        trainloader=trainloader,
        testloader=testloader,
        diffusion=diffusion,
        noise_steps=noise_steps,
        position_encoding_dim=position_encoding_dim,
        positional_encoding=positional_encoding,
        num_epochs=epochs,
        logger=logger,
        logpath=logpath,
        train_msg=train_msg,
        EVALUATE_METRICS=EVALUATE_METRICS,
        mae_metric=mae_metric,
        ms_ssim_metric=ms_ssim_metric,
        ssim_metric=ssim_metric,
        psnr_metric=psnr_metric,
        train_loss=train_loss,
        mae_results=mae_results,
        ms_ssim_results=ms_ssim_results,
        ssim_results=ssim_results,
        psnr_results=psnr_results,
        nrmse_results=nrmse_results,
    )

    logger.success("✅ 模型训练完成（Step 2-4）")

    # ---- 2-5 Save the model and plot the loss ----
    savepath = ADDR_ROOT / "saves"
    loss_plot_path = savepath / "TRAIN" / f"{model_save_name}.png"
    loss_data_path = savepath / "TRAIN" / f"{model_save_name}.npy"
    model_save_folder = savepath / "MODEL"
    
    torch.save(unet.state_dict(), f'{model_save_folder}/unetconfig_{model_save_name}.pth')
    diffusion_config = {
        'noise_steps': diffusion.noise_steps,
        'beta_start': diffusion.beta_start,
        'beta_end': diffusion.beta_end,
        'img_size': diffusion.img_size,
    }
    torch.save(diffusion_config, f'{model_save_folder}/diffusionconfig_{model_save_name}.pth')
    
    # 训练过程图
    plt.figure()
    plt.plot(train_loss, "g-o", label="Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(loss_plot_path, dpi=300)

    palette = sns.color_palette("Dark2")
    fig, ax = plt.subplots(1, 3, figsize=(19,5))
    ax[0].plot(mae_results, color=palette[0], marker="o", label="MAE")
    ax[0].plot(nrmse_results, color=palette[1], marker="o", label="NRMSE")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("MAE / NRMSE")
    ax[0].legend()

    ax[1].plot(ms_ssim_results, color=palette[3], marker="o", label="MS-SSIM")
    ax[1].plot(ssim_results, color=palette[4], marker="o", label="SSIM")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("MS-SSIM / SSIM")
    ax[1].legend()

    ax[2].plot(psnr_results, color=palette[5], marker="o", label="PSNR")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("PSNR")
    ax[2].legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)


    logger.success(f"Loss plot saved at {loss_plot_path}")
    logger.success(f"Loss data saved at {loss_data_path}")
    logger.success(f"Model saved at {model_save_folder}")
    logger.success("✅ 模型保存完成（Step 2-5）")
    # -----------------------------------------
if __name__ == "__main__":
    app()
