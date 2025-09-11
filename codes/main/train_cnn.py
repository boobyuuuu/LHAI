# 所有CNN类模型通用训练框架

# ---- 01 Improt Libraries ----
# ---- 1-1 Path and Logging ----
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
# ---- 1-2 Configuration and Modules ----
from codes.function.Log import log
import codes.function.Train as Train
import codes.function.Loss as lossfunction
from codes.config.config_cnn import TrainConfig
from codes.config.config_cnn import ModelConfig
from codes.function.Dataset import ImageDataset, DataModule
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
    model_name: str = train_cfg.model_name,              # para02：模型名称 default: "CNN"
    seed: int = train_cfg.seed,                          # para08：随机种子 default: 0
    frac: float = train_cfg.frac,                        # para09：训练集比例 default: 0.8
    epochs: int = train_cfg.epochs,                      # para10：训练轮数 default: 400
    batch_size: int = train_cfg.batch_size,              # para11：批次大小 default: 32
    lr_max: float = train_cfg.lr_max,                    # para12：最大学习率 default: 5e-4
    lr_min: float = train_cfg.lr_min,                    # para13：最小学习率 default: 5e-6
    datarange: float = train_cfg.datarange,               # para14：数据范围 default: 1.0
):
    # ==== 2-1 Initialization  ====
    # train 自定义参数
    data_path = data_dir / data_name
    model_path = model_dir / f"{model_name}.py"

    # exp 实验参数
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # save path
    dataname = data_name.split("_")[0]
    model_save_name = f"{model_name}_{exp_name}_{epochs}epo_{batch_size}bth_{dataname}"

    save_dir_train = ADDR_ROOT / "saves" / "TRAIN" / model_name
    save_dir_model = ADDR_ROOT / "saves" / "MODEL" / model_name
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    log_dir = save_dir_train
    logpath = log_dir / f"trainlog_{model_name}"

    loss_plot_path = save_dir_train / f"{model_save_name}.png"
    loss_data_path = save_dir_train / f"{model_save_name}.npy"

    Best_model_save_path = save_dir_model / f"Best_{model_save_name}.pth"
    Last_model_save_path = save_dir_model / f"Last_{model_save_name}.pth"

    logger.success("========= 2-1 参数加载完成 =========")
    
    # ==== 2-2 Data: trainloader & testloader ====
    dm = DataModule(
        data_path=data_path,
        batch_size=batch_size,
        frac=frac,
        inverse=False,
        shuffle_train=False,
        shuffle_test=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    trainloader, testloader = dm.build()

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

    # ==== 2-3 Initialize the model, loss function and optimizer ====
    # model
    model_params = model_cfg.model_params
    params = model_params[model_name]

    sys.path.append(str(model_dir))
    module = importlib.import_module(model_name)
    MODEL = getattr(module, model_name)

    model = MODEL(**params).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)
    lr_lambda = lambda epoch: lr_min / lr_max + 0.5 * (1 - lr_min / lr_max) * (1 + np.cos(np.pi * epoch / epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # loss function
    criterion = lossfunction.msejsloss

    logger.success("========= 2-3 模型、损失函数、优化器加载完成 =========")

    # ==== 2-4 Initialize the training function ====
    train = Train.train_cnn

    # logger output
    format_model_params = Train.format_model_params
    torch.set_printoptions(precision=10)
    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    optimizer_name = optimizer.__class__.__name__                      # 优化器类名，例如 AdamW
    loss_name = criterion.__name__                                  # 损失函数名称，例如 msejsloss
    model_params_str = format_model_params(model_params[model_name])
    filetmp = np.load(data_path, allow_pickle=True)
    filelen = int(filetmp.shape[0])
    del filetmp
    train_msg = f"""
    ====================== 训练参数 ======================
    🔧 配置信息概览：
    - traintime               : {train_time}
    - exp_name                : {exp_name}
    - model_name              : {model_name}
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
    - model_params            :

    {model_params_str}
    ==============================================================
    """
    logger.info(train_msg)

    LOSS_PLOT = []
    TESTLOSS_PLOT = []
    EPOCH_PLOT = []

    train(
        model=model,
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
        Best_model_save_path=Best_model_save_path)
    logger.success(f"========= 2-4 模型训练完成, 训练log已保存在{logpath} =========")

    # ==== 2-5 Save model and loss ====
    # model save
    torch.save(model.state_dict(), Last_model_save_path)
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
    model.eval()
    model.to(device)
    for _, (img_LR, img_HR) in enumerate(testloader):
        img_SR, _, _ = model(img_LR.to(device))
        img_SR = img_SR.cpu()
        break
    num_images_to_show = 8
    fig, axes = plt.subplots(num_images_to_show,5 , figsize=(15, 3 * num_images_to_show))
    for i in range(num_images_to_show):
        blurry_img_numpy = img_LR[i].squeeze().detach().cpu().numpy()
        sr_img_numpy = img_SR[i].squeeze().detach().cpu().numpy()
        original_img_numpy = img_HR[i].squeeze().detach().cpu().numpy()
        blurry_img_numpy =blurry_img_numpy/blurry_img_numpy.sum()
        original_img_numpy=original_img_numpy/original_img_numpy.sum()
        sr_img_numpy =sr_img_numpy/sr_img_numpy.sum() 
        
        im0=axes[i, 0].imshow(blurry_img_numpy)
        axes[i, 0].set_title('Blurry Image')
        axes[i, 0].axis('off')

        im1=axes[i, 1].imshow(sr_img_numpy)
        axes[i, 1].set_title('SR Image')
        axes[i, 1].axis('off')
        
        im2=axes[i, 2].imshow(original_img_numpy)
        axes[i, 2].set_title('Original Image')
        axes[i, 2].axis('off')

        res_blur = (blurry_img_numpy-original_img_numpy)
        res_sr = (sr_img_numpy-original_img_numpy)
        vmin = min(res_blur.min(),res_sr.min())
        vmax = max(res_blur.max(),res_sr.max())

        im3= axes[i, 3].imshow(res_blur,vmin =vmin,vmax=vmax)
        axes[i, 3].set_title('Res Blur')
        axes[i, 3].axis('off')

        im4 = axes[i, 4].imshow(res_sr,vmin =vmin,vmax=vmax)
        axes[i, 4].set_title('Res SR')
        axes[i, 4].axis('off')
        cbar2 = fig.colorbar(
            im4, ax=axes[i,4],shrink = 0.5
        )
    plt.tight_layout()
    plt.savefig(f'{save_dir_train}/Trainpredict_{model_save_name}.png', dpi = 300)
    plt.show()
    logger.success(f"First prediction saved at {save_dir_train}/Trainpredict_{model_save_name}.png")
    logger.success("========= 2-6 模型初次推理完成 =========")
    # -----------------------------------------
if __name__ == "__main__":
    app()
