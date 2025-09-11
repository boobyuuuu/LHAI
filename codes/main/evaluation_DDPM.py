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
from codes.config.config_DDPM import EvalConfig
from codes.config.config_DDPM import ModelConfig
from codes.function.Dataset import ImageDataset, DataModule
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
eval_cfg = EvalConfig()
model_cfg = ModelConfig()
app = typer.Typer()
@app.command()
def main(
    exp_name: str = eval_cfg.exp_name,                      # para01：实验名称 default: "EXP01"
    data_dir: Path = eval_cfg.data_dir,                 # para05：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = eval_cfg.data_name,                # para06：数据文件名 default: "xingwei_10000_64_train_v1.npy"
    model_dir: Path = eval_cfg.model_dir,               # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_name_diffusion: str = eval_cfg.model_name_diffusion,              # para02：模型名称 default: "DIFFUSION"
    model_name_unet: str = eval_cfg.model_name_unet,                            # para02：模型名称 default: "UNET"
    seed: int = eval_cfg.seed,                              # para11：随机种子 default: 0
    frac: float = eval_cfg.frac,                            # para12：训练集比例 default: 0.8
    batch_size: int = eval_cfg.batch_size,                  # para13：批次大小 default: 32
    epochs: int = eval_cfg.epochs,                          # para14：训练轮数（如需评估多个 epoch） default: 400
    lr_max: float = eval_cfg.lr_max,                        # para15：最大学习率 default: 5e-4
    lr_min: float = eval_cfg.lr_min,                        # para16：最小学习率 default: 5e-6
    datarange: float = eval_cfg.datarange,
    LoB: str = eval_cfg.LoB,                                # para17：选择加载Best还是Last模型 default: "Last"
    dataname: str = eval_cfg.dataname                       # para18：数据集名称缩写 default: "xingwei"
):
    # ==== 2-1 Initialization  ====
    # train 自定义参数
    data_path = data_dir / data_name
    model_path = model_dir / f"{model_name_diffusion}.py"
    model_weight_name = f"{LoB}_{model_name_diffusion}_{exp_name}_{epochs}epo_{batch_size}bth_{dataname}.pth"
    model_weight_dir = ADDR_ROOT / "saves" / "MODEL" / model_name_diffusion
    model_weight_path = model_weight_dir / model_weight_name

    # exp 实验参数
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    
    # ==== 2-4 Evaluation ====

    # save path
    dataname = data_name.split("_")[0]
    save_dir_eval = ADDR_ROOT / "saves" / "EVAL" / model_name_diffusion
    if not os.path.exists(save_dir_eval):
        os.makedirs(save_dir_eval)

    # logger output
    format_model_params = Train.format_model_params
    torch.set_printoptions(precision=10)
    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    loss_name = trainingloss.__name__
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
    - device                  : {device}({gpu_name})
    - model_params_diffusion  : 
    
    {model_params_str_diffusion}
    - model_params_unet       :

    {model_params_str_unet}
    ==============================================================
    """
    logger.info(train_msg)

    # ---- 2-3 evaluation 2: lineprofiles and resmap----
    def interp2d(x1,x2,y1,y2,arr):
        x = np.arange(arr.shape[0])
        y = np.arange(arr.shape[0])
        xx,yy = np.meshgrid(x,y)
        interpolate = scipy.interpolate.RegularGridInterpolator((x,y),arr)
        y_t = np.linspace(x1,x2,101)
        x_t = np.linspace(y1,y2,101)
        z_t = interpolate((x_t,y_t))
        return z_t
    
    # 获取测试集中的第一批图像
    test_input_images, test_target_images = next(iter(testloader))
    test_input_images = test_input_images.to(device)
    test_target_images = test_target_images.to(device)
    
    n_images = 10 # 显示前n张图像

    # 推理（生成高分辨图像序列）
    with torch.no_grad():
        generated_images = diffusion.reverse_diffusion(
            model=unet,
            n_images=n_images,
            n_channels=1,
            input_image=test_input_images[:n_images],
            save_time_steps=None
        )

    # 转置维度，从 (T, B, C, H, W) → (B, T, C, H, W)
    generated_images = generated_images.swapaxes(0, 1)
    
    showlist = [0,1,2,3,4,5]# 你可以改这个，展示哪几张图
    num_images_to_show = len(showlist)
    xys = np.zeros(num_images_to_show).tolist()
    for i in range(num_images_to_show):
        xys[i] = [0,0,0,0]
    xys[0] = [10, 50, 37, 37]
    xys[1] = [10, 40, 20, 5]
    xys[2] = [40, 50, 20, 10]
    xys[3] = [20, 40, 40, 23]
    xys[4] = [20, 50, 50, 40]
    xys[5] = [20, 60, 60, 20]
    
    # 提取图像：低分辨图 / 生成图 / 原始图
    img_LR = test_input_images.cpu()
    img_HR = test_target_images.cpu()

    fig, axes = plt.subplots(num_images_to_show, 6, figsize=(18, 3 * num_images_to_show))

    for i in range(num_images_to_show):
        count = showlist[i]
        x1, x2, y1, y2 = xys[i]
        color = 'white'

        # 提取图像
        blurry_img_numpy = img_LR[count].squeeze().numpy()
        original_img_numpy = img_HR[count].squeeze().numpy()

        image_diff_trajectory = generated_images[:, count]
        sr_img_numpy = image_diff_trajectory[-1].cpu()
        sr_img_numpy = sr_img_numpy.squeeze().numpy()

        # 归一化（可选）
        blurry_img_numpy = blurry_img_numpy / blurry_img_numpy.sum()
        sr_img_numpy = sr_img_numpy / sr_img_numpy.sum()
        original_img_numpy = original_img_numpy / original_img_numpy.sum()

        # Blurry 图像
        axes[i, 0].imshow(blurry_img_numpy, cmap='viridis')
        axes[i, 0].set_title('Blurry Image')
        axes[i, 0].plot([x1, x2], [y1, y2], linestyle='--', color=color, linewidth=2)
        axins = inset_axes(axes[i, 0], width="20%", height="20%", loc=4)
        axins.axis('off')
        axins.patch.set_alpha(0)
        axins.plot(interp2d(x1, x2, y1, y2, blurry_img_numpy), color=color)

        # SR 图像
        axes[i, 1].imshow(sr_img_numpy, cmap='viridis')
        axes[i, 1].set_title('SR Image')
        axes[i, 1].plot([x1, x2], [y1, y2], linestyle='--', color=color, linewidth=2)
        axins = inset_axes(axes[i, 1], width="20%", height="20%", loc=4)
        axins.axis('off')
        axins.patch.set_alpha(0)
        axins.plot(interp2d(x1, x2, y1, y2, sr_img_numpy), color=color)

        # 原始 HR 图像
        axes[i, 2].imshow(original_img_numpy, cmap='viridis')
        axes[i, 2].set_title('Original Image')
        axes[i, 2].plot([x1, x2], [y1, y2], linestyle='--', color=color, linewidth=2)
        axins = inset_axes(axes[i, 2], width="20%", height="20%", loc=4)
        axins.axis('off')
        axins.patch.set_alpha(0)
        axins.plot(interp2d(x1, x2, y1, y2, original_img_numpy), color=color)

        # 残差图
        res_blur = blurry_img_numpy - original_img_numpy
        res_sr = sr_img_numpy - original_img_numpy
        vmin = min(res_blur.min(), res_sr.min())
        vmax = max(res_blur.max(), res_sr.max())

        im3 = axes[i, 3].imshow(res_blur, vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 3].set_title('Res Blur')
        fig.colorbar(im3, ax=axes[i, 3], shrink=0.5)

        im4 = axes[i, 4].imshow(res_sr, vmin=vmin, vmax=vmax, cmap='viridis')
        axes[i, 4].set_title('Res SR')
        fig.colorbar(im4, ax=axes[i, 4], shrink=0.5)

        # 曲线图
        axes[i, 5].plot(interp2d(x1, x2, y1, y2, blurry_img_numpy), color='red', label='Blurry')
        axes[i, 5].plot(interp2d(x1, x2, y1, y2, sr_img_numpy), color='blue', label='SR')
        axes[i, 5].plot(interp2d(x1, x2, y1, y2, original_img_numpy), color='black', label='Original')
        axes[i, 5].set_title('Line Profiles')
        axes[i, 5].legend()
    
    plt.tight_layout()
    savepath = f'{ADDR_ROOT}/saves'
    savefigname = f"Eval_distribution_{model_weight_name}"
    savefig2_path = f'{save_dir_eval}/{savefigname}.png'
    plt.savefig(savefig2_path, dpi=300)
    logger.success(f"Evaluation 1: Loss figure saved at {savefig2_path}")
    logger.success("========= 2-4-2 lineprofiles and resmap 评估完成 =========")
    
    # ---- 2-3 evaluation 3: NRMSE,MAE,MS-SSIM,SSIM,PSNR ----
    ms_ssim_metric = MS_SSIM(
        data_range=2.0, kernel_size=7, betas=(0.0448, 0.2856, 0.3001)
    ).to(device)
    ssim_metric = SSIM(data_range=2.0).to(device)
    psnr_metric = PSNR(data_range=2.0).to(device)
    mae_metric = MAE().to(device)

    # 你已有的测试输入图像
    test_input_images, test_target_images = next(iter(testloader))
    test_input_images = test_input_images.to(device)
    test_target_images = test_target_images.to(device)

    n_images = 24  # 评估前n张图像

    # 推理生成高分辨图像序列
    with torch.no_grad():
        generated_images = diffusion.reverse_diffusion(
            model=unet,
            n_images=n_images,
            n_channels=1,
            input_image=test_input_images[:n_images],
            save_time_steps=None
        )

    # 转置维度 (T, B, C, H, W) → (B, T, C, H, W)
    # generated_images = generated_images.swapaxes(0, 1)

    # 存储每张图像的评估指标
    nrmse_list, mae_list, ms_ssim_list, ssim_list, psnr_list = [], [], [], [], []
    nrmse_ipt_list, mae_ipt_list, ms_ssim_ipt_list, ssim_ipt_list, psnr_ipt_list = [], [], [], [], []

    for i in range(n_images):
        image_diff_trajectory = generated_images[:, i]  # shape: (T, C, H, W)
        generated_high_res_image = image_diff_trajectory[-1]  # 最后一帧
        target_high_res_image = test_target_images[i]
        input_low_res_image = test_input_images[i]

        # 将图像转到 CPU
        gen_img = generated_high_res_image.cpu()
        tgt_img = target_high_res_image.cpu()
        ipt_img = input_low_res_image.cpu()

        # diff = tgt_img - ipt_img
        # diff = tgt_img - gen_img
        # np.save(f'diff_gen{i}.npy',diff)

        # 指标计算 -- gen
        mae = torch.mean(torch.abs(gen_img - tgt_img)).item()
        nrmse = torch.sqrt(torch.mean((gen_img - tgt_img) ** 2)) / (tgt_img.max() - tgt_img.min())
        nrmse = nrmse.item()

        mae_list.append(mae)
        nrmse_list.append(nrmse)

        ms_ssim_val = ms_ssim_metric(gen_img.unsqueeze(0), tgt_img.unsqueeze(0)).item()
        ssim_val    = ssim_metric(gen_img.unsqueeze(0), tgt_img.unsqueeze(0)).item()
        psnr_val    = psnr_metric(gen_img.unsqueeze(0), tgt_img.unsqueeze(0)).item()

        ms_ssim_list.append(ms_ssim_val)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

        # 指标计算 -- input
        mae_ipt = torch.mean(torch.abs(ipt_img - tgt_img)).item()
        nrmse_ipt = torch.sqrt(torch.mean((ipt_img - tgt_img) ** 2)) / (tgt_img.max() - tgt_img.min())
        nrmse_ipt = nrmse_ipt.item()

        mae_ipt_list.append(mae_ipt)
        nrmse_ipt_list.append(nrmse_ipt)

        ms_ssim_ipt_val = ms_ssim_metric(ipt_img.unsqueeze(0), tgt_img.unsqueeze(0)).item()
        ssim_ipt_val    = ssim_metric(ipt_img.unsqueeze(0), tgt_img.unsqueeze(0)).item()
        psnr_ipt_val    = psnr_metric(ipt_img.unsqueeze(0), tgt_img.unsqueeze(0)).item()

        ms_ssim_ipt_list.append(ms_ssim_ipt_val)
        ssim_ipt_list.append(ssim_ipt_val)
        psnr_ipt_list.append(psnr_ipt_val)

    # 输出平均指标
    print("\n=== Average Metrics on {} Test Images ===".format(n_images))
    print("NRMSE:   {:.6f}".format(np.mean(nrmse_list)))
    print("MAE:     {:.6f}".format(np.mean(mae_list)))
    print("MS-SSIM: {:.6f}".format(np.mean(ms_ssim_list)))
    print("SSIM:    {:.6f}".format(np.mean(ssim_list)))
    print("PSNR:    {:.6f}".format(np.mean(psnr_list)))

    print("\n=== Average Metrics on {} Input Images ===".format(n_images))
    print("NRMSE:   {:.6f}".format(np.mean(nrmse_ipt_list)))
    print("MAE:     {:.6f}".format(np.mean(mae_ipt_list)))
    print("MS-SSIM: {:.6f}".format(np.mean(ms_ssim_ipt_list)))
    print("SSIM:    {:.6f}".format(np.mean(ssim_ipt_list)))
    print("PSNR:    {:.6f}".format(np.mean(psnr_ipt_list)))


    # ==== 作图 ====
    palette = sns.color_palette("Dark2")
    image_ids = list(range(1, n_images + 1))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))

    # 1. NRMSE & MAE
    ax[0].plot(image_ids, mae_list, color=palette[0], marker='s', label="MAE")
    ax[0].plot(image_ids, nrmse_list, color=palette[1], marker='o', label="NRMSE")
    ax[0].plot(image_ids, mae_ipt_list, color=palette[3], marker='s', label="MAE_ipt")
    ax[0].plot(image_ids, nrmse_ipt_list, color=palette[4], marker='o', label="NRMSE_ipt")
    ax[0].set_xlabel("Image Index")
    ax[0].set_ylabel("Value")
    ax[0].set_title("NRMSE & MAE per Image")
    ax[0].legend()
    ax[0].grid(True)

    # 2. MS-SSIM & SSIM
    ax[1].plot(image_ids, ms_ssim_list, color=palette[3], marker='o', label="MS-SSIM")
    ax[1].plot(image_ids, ssim_list, color=palette[4], marker='s', label="SSIM")
    ax[1].plot(image_ids, ms_ssim_ipt_list, color=palette[5], marker='o', label="MS-SSIM_ipt")
    ax[1].plot(image_ids, ssim_ipt_list, color=palette[6], marker='s', label="SSIM_ipt")
    ax[1].set_xlabel("Image Index")
    ax[1].set_ylabel("Value")
    ax[1].set_title("MS-SSIM & SSIM per Image")
    ax[1].legend()
    ax[1].grid(True)

    # 3. PSNR
    ax[2].plot(image_ids, psnr_list, color=palette[5], marker='o', label="PSNR_ipt")
    ax[2].plot(image_ids, psnr_ipt_list, color=palette[7], marker='o', label="PSNR_ipt")
    ax[2].set_xlabel("Image Index")
    ax[2].set_ylabel("Value")
    ax[2].set_title("PSNR per Image")
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout()
    
    savefig3_path = f'{save_dir_eval}/Eval_metrics_{model_weight_name}.png'
    plt.savefig(savefig3_path)
    plt.close()
    logger.info(f"Evaluation plots saved at {savefig3_path}")
    logger.success("========= 2-4-3 NRMSE, MAE, MS-SSIM, SSIM, PSNR 评估完成 =========")
    # -----------------------------------------

if __name__ == "__main__":
    app()
