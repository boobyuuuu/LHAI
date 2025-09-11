# 所有DIFFUSION方案模型通用训练框架

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
from codes.config.config_diffusion import TrainConfig
from codes.config.config_diffusion import ModelConfig
from codes.function.Dataset import ImageDataset
# ---- 1-3 Libraries for pytorch and others ----
# ---- 1-3 PyTorch ----
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split, ConcatDataset
# ---- 1-4 Others ----
import importlib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


import deeplay as dl
# ---- 02 Define the main function ----
train_cfg = TrainConfig()
model_cfg = ModelConfig()
app = typer.Typer()
@app.command()
def main(
    exp_name: str = train_cfg.exp_name,                  # para01：实验名称 default: "EXP01"
    data_dir: Path = train_cfg.data_dir,                 # para05：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = train_cfg.data_name,                # para06：数据文件名 default: "xingwei_10000_64_train_v1.npy"
    model_dir: Path = train_cfg.model_dir,               # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_name_diffusion: str = train_cfg.model_name_diffusion,              # para02：模型名称 default: "DIFFUSION"
    model_name_unet: str = train_cfg.model_name_unet,                            # para02：模型名称 default: "UNET"
    seed: int = train_cfg.seed,                          # para08：随机种子 default: 0
    frac: float = train_cfg.frac,                        # para09：训练集比例 default: 0.8
    epochs: int = train_cfg.epochs,                      # para10：训练轮数 default: 400
    batch_size: int = train_cfg.batch_size,              # para11：批次大小 default: 32
    lr_max: float = train_cfg.lr_max,                    # para12：最大学习率 default: 5e-4
    lr_min: float = train_cfg.lr_min,                    # para13：最小学习率 default: 5e-6
    datarange: float = train_cfg.datarange,              # para14：数据范围 default: 1.0
):
    # ==== 2-1 Initialization  ====
    # train 自定义参数
    data_path = data_dir / data_name
    model_path = model_dir / f"{model_name_diffusion}.py"

    # exp 实验参数
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # save path
    dataname = data_name.split("_")[0]
    model_save_name = f"{model_name_diffusion}_{exp_name}_{epochs}epo_{batch_size}bth_{dataname}"

    save_dir_train = ADDR_ROOT / "saves" / "TRAIN" / model_name_diffusion
    save_dir_model = ADDR_ROOT / "saves" / "MODEL" / model_name_diffusion
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    log_dir = save_dir_train
    logpath = log_dir / f"trainlog_{model_name_diffusion}"

    loss_plot_path = save_dir_train / f"{model_save_name}.png"
    loss_data_path = save_dir_train / f"{model_save_name}.npy"

    Best_model_save_path = save_dir_model / f"Best_{model_save_name}.pth"
    Last_model_save_path = save_dir_model / f"Last_{model_save_name}.pth"

    logger.success("========= 2-1 参数加载完成 =========")
    
    # ==== 2-2 Data: trainloader & testloader ====
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
    ====================== 数据参数 ======================
    Output of data from Batch 1

    - blurry image     : {blurry_img_shape} [批次, 通道数, 高度, 宽度]
    - clear image      : {original_img_shape} [批次, 通道数, 高度, 宽度]
    - datarange        : 最小值 = {blurry_img_min:.6f}, 最大值 = {blurry_img_max:.6f}
    - 1st image output :

    {np.array2string(blurry_img_sample, precision=4, suppress_small=True, threshold=64)}
    ===============================================================
    """)

    logger.success("========= 2-2 数据加载完成 =========")

    # ---- 2-3 Initialize the model, loss function and optimizer ----
    # DIFFUSION
    model_params = model_cfg.model_params
    params_diffusion = model_params[model_name_diffusion]
    params_unet = model_params[model_name_unet]

    sys.path.append(str(model_dir))
    module_diffusion = importlib.import_module(model_name_diffusion)
    DIFFUSION = getattr(module_diffusion, model_name_diffusion)

    diffusion = DIFFUSION(**params_diffusion).to(device)
    # Unet
    model_params = model_cfg.model_params
    params_unet = model_params[model_name_unet]
    sys.path.append(str(model_dir))
    module_unet = importlib.import_module(model_name_unet)
    UNET = getattr(module_unet, model_name_unet)

    unet = UNET(**params_unet).to(device)
    unet.build()
    unet.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr_max)
    lr_lambda = lambda epoch: lr_min / lr_max + 0.5 * (1 - lr_min / lr_max) * (1 + np.cos(np.pi * epoch / epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # loss function
    criterion = lossfunction.msejsloss

    logger.success("========= 2-3 模型、损失函数、优化器加载完成 =========")

    # ==== 2-4 Initialize the training function ====
    train = Train.train_diffusion  # 确认train_diffusion函数签名与之前定义的train一致

    # logger output
    format_model_params = Train.format_model_params
    torch.set_printoptions(precision=10)
    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    optimizer_name = optimizer.__class__.__name__                      # 优化器类名，例如 AdamW
    loss_name = criterion.__name__                                  # 损失函数名称，例如 msejsloss
    model_params_str_diffusion = format_model_params(model_params[model_name_diffusion])
    model_params_str_unet = format_model_params(model_params[model_name_unet])
    train_msg = f"""
    ====================== 训练参数 ======================
    🔧 配置信息概览：
    - traintime               : {train_time}
    - exp_name                : {exp_name}
    - model_name              : {model_name_diffusion} + {model_name_unet}
    - data_name               : {data_name}（{dataname}）
    - model_path              : {model_path}
    - data_path               : {data_path}
    - seed                    : {seed}
    - frac                    : 训练集 {frac*100:.1f}% / 测试集 {100-frac*100:.1f}%
    - datalength              : {filelen}
    - epochs                  : {epochs}
    - batch_size              : {batch_size}
    - datarange               : {datarange}
    - learnrate               : 最小 = {lr_min:.1e}, 最大 = {lr_max:.1e}
    - lossname                : {loss_name}
    - optimizer               : {optimizer_name}
    - device                  : {device}({gpu_name})
    - logpath                 : {logpath}
    - model_params_diffusion  : 
    
    {model_params_str_diffusion}
    - model_params_unet       :

    {model_params_str_unet}
    ==============================================================
    """
    logger.info(train_msg)

    LOSS_PLOT = []
    TESTLOSS_PLOT = []
    EPOCH_PLOT = []

    train(
        unet=unet,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=epochs,
        logger=logger,
        logpath=logpath,
        train_msg=train_msg,
        LOSS_PLOT=LOSS_PLOT,
        TESTLOSS_PLOT=TESTLOSS_PLOT,
        EPOCH_PLOT=EPOCH_PLOT,
        Best_model_save_path=Best_model_save_path,
        grad_clip=1.0,                 # 梯度裁剪，防止梯度爆炸
        eval_sample_every=None,        # e.g., 10 开启小样本采样
        save_time_steps=None           # e.g., [100, 500, 999]
    )
    logger.success(f"========= 2-4 模型训练完成, 训练log已保存在{logpath} =========")

    # ==== 2-5 Save model and loss ====
    torch.save(unet.state_dict(), Last_model_save_path)
    logger.info(f"Last model saved at {Last_model_save_path}")

    # loss plot save
    fig, ax = plt.subplots()
    ax.plot(EPOCH_PLOT, LOSS_PLOT)
    ax.plot(EPOCH_PLOT, TESTLOSS_PLOT)
    ax.set_yscale('log')
    fig.savefig(loss_plot_path, dpi=300)
    logger.info(f"Loss plot saved at {loss_plot_path}")

    # loss data save
    LOSS_DATA = np.stack((np.array(EPOCH_PLOT), np.array(LOSS_PLOT), np.array(TESTLOSS_PLOT)), axis=0)
    np.save(loss_data_path, LOSS_DATA)
    logger.info(f"Loss data saved at {loss_data_path}")

    logger.success("========= 2-5 模型保存完成 =========")

    # ==== 2-6 First prediction ====
    unet.eval()
    unet.to(device)
    with torch.no_grad():
        for _, (img_LR, img_HR) in enumerate(testloader):
            img_LR = img_LR.to(device)     # (B, 1, 64, 64) 条件输入
            img_HR = img_HR.to(device)     # (B, 1, 64, 64) 仅用于可视化对比
            break
    # 采样（反向扩散）
    # - n_images: 本批数量
    # - n_channels: 输出通道（与你数据一致：1）
    # - input_image: 条件图像（LR）
    # - save_time_steps: 若你想保存中间过程，可传列表，如 [999, 500, 100]
    with torch.no_grad():
        samples = diffusion.reverse_diffusion(
            model=unet,
            n_images=img_LR.size(0),
            n_channels=img_LR.size(1),   # =1
            input_image=img_LR,          # 条件输入
            save_time_steps=None         # 或者 [100, 500, 999]
        )
    # 兼容两种返回：如果 reverse_diffusion 返回的是中间快照堆叠（B, K, C, H, W）或者只返回最终结果（B, C, H, W）
    if samples.dim() == 5:
        # 只取最后一个时间点（通常 save_time_steps 最小的那个，但这里保守取最后一张）
        img_SR = samples[:, -1, ...].contiguous().cpu()
    else:
        img_SR = samples.contiguous().cpu()   # (B, 1, 64, 64)

    img_LR_cpu = img_LR.detach().cpu()
    img_HR_cpu = img_HR.detach().cpu()

    num_images_to_show = min(8, img_SR.size(0))
    fig, axes = plt.subplots(num_images_to_show, 5, figsize=(15, 3 * num_images_to_show))
    for i in range(num_images_to_show):
        blurry_img_numpy   = img_LR_cpu[i].squeeze().numpy()
        sr_img_numpy       = img_SR[i].squeeze().numpy()
        original_img_numpy = img_HR_cpu[i].squeeze().numpy()

        eps = 1e-12
        blurry_img_numpy   = blurry_img_numpy / max(blurry_img_numpy.sum(), eps)
        original_img_numpy = original_img_numpy / max(original_img_numpy.sum(), eps)
        sr_img_numpy       = sr_img_numpy / max(sr_img_numpy.sum(), eps)

        im0 = axes[i, 0].imshow(blurry_img_numpy)
        axes[i, 0].set_title('Blurry Image')
        axes[i, 0].axis('off')

        im1 = axes[i, 1].imshow(sr_img_numpy)
        axes[i, 1].set_title('SR (Diffusion)')
        axes[i, 1].axis('off')

        im2 = axes[i, 2].imshow(original_img_numpy)
        axes[i, 2].set_title('Original Image')
        axes[i, 2].axis('off')

        res_blur = blurry_img_numpy - original_img_numpy
        res_sr   = sr_img_numpy - original_img_numpy
        vmin = min(res_blur.min(), res_sr.min())
        vmax = max(res_blur.max(), res_sr.max())

        im3 = axes[i, 3].imshow(res_blur, vmin=vmin, vmax=vmax)
        axes[i, 3].set_title('Res Blur')
        axes[i, 3].axis('off')

        im4 = axes[i, 4].imshow(res_sr, vmin=vmin, vmax=vmax)
        axes[i, 4].set_title('Res SR')
        axes[i, 4].axis('off')
        fig.colorbar(im4, ax=axes[i, 4], shrink=0.5)

    plt.tight_layout()
    plt.savefig(f'{save_dir_train}/Trainpredict_{model_save_name}.png', dpi=300)
    plt.show()

    logger.success(f"First prediction saved at {save_dir_train}/Trainpredict_{model_save_name}.png")
    logger.success("========= 2-6 Diffusion 模型初次推理完成 =========")
    # -----------------------------------------
if __name__ == "__main__":
    app()
