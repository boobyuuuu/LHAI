# 所有DIFFUSION方案模型通用测试框架

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
# ---- 1-2 Libraries for Configuration ----
from codes.config.config_diffusion import EvalConfig
from codes.function.Dataset import ImageDataset
import codes.function.Loss as lossfunction
from codes.function.Log import log
import codes.function.Train as Train
from codes.models.DIFFUSION_EXP01 import EnhancedUNetWrapper
from codes.models.DIFFUSION_EXP01 import Diffusion
from codes.models.DIFFUSION_EXP01 import positional_encoding
from codes.models.DIFFUSION_EXP01 import prepare_data
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
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanAbsoluteError as MAE
import scipy.interpolate

app = typer.Typer()

# ---- 02 Define the main function ----
eval_cfg = EvalConfig()
@app.command()
def main(
    exp_name: str = eval_cfg.exp_name,                      # para01：实验名称 default: "EXP01"
    model_name: str = eval_cfg.model_name,                  # para02：模型名称 default: "CNN"
    model_dir: Path = eval_cfg.model_dir,                   # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_path: Path = eval_cfg.model_path,                 # para04：模型定义路径（.py） default: model_dir / f"{model_name}_{exp_name}.py"
    data_dir: Path = eval_cfg.data_dir,                   # para05：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = eval_cfg.data_name,                    # para06：数据名称 default: "xingwei_10000_64_train_v1.npy"
    data_path: Path = eval_cfg.data_path,                   # para07：数据路径 default: data_dir / data_name
    unet_weight_name: str = eval_cfg.unet_weight_name,      # para08：UNet权重名称 default: "unetconfig_DIFFUSION_EXP01_1epo_32bth_xingwei.pth"
    unet_weight_path: Path = eval_cfg.unet_weight_path,      # para09：UNet权重路径 default: ADDR_ROOT / "saves" / "MODEL" / unet_weight_name
    diffusion_weight_name: str = eval_cfg.diffusion_weight_name,  # para10：Diffusion权重名称 default: "diffusionconfig_DIFFUSION_EXP01_1epo_32bth_xingwei.pth"
    diffusion_weight_path: Path = eval_cfg.diffusion_weight_path,  # para11：Diffusion权重路径 default: ADDR_ROOT / "saves" / "MODEL"/ diffusion_weight_name
    seed: int = eval_cfg.seed,                              # para12：随机种子 default: 0
    frac: float = eval_cfg.frac,                            # para13：数据集划分比例 default: 0.98
    epochs: int = eval_cfg.epochs,                          # para14：训练轮数 default: 1
    batch_size: int = eval_cfg.batch_size,                  # para15：批次大小 default: 32
    lr_max: float = eval_cfg.lr_max,                        # para16：最大学习率 default: 5e-4
    lr_min: float = eval_cfg.lr_min,                        # para17：最小学习率 default: 5e-6
    datarange: float = eval_cfg.datarange,                  # para18：数据范围 default: 1.0
    position_encoding_dim: int = eval_cfg.position_encoding_dim,  # para19：位置编码维度 default: 256
    noise_steps: int = eval_cfg.noise_steps,                # para20：噪声步骤 default: 2000
    logpath: Path = eval_cfg.logpath                        # para21：日志路径 default: ADDR_ROOT / "logs" / "train_diffusion.log"
):
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
    logger.success("✅ 参数加载完成（Step 2-1）")
    
    # ---- 2-2 Load the data and model ----
    filetmp = np.load(data_path,allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    NUM_TO_LEARN = int(filelen)

    dataset = ImageDataset(NUM_TO_LEARN,data_path,inverse=False)
    trainset, testset = random_split(dataset,
        lengths=[int(frac *len(dataset)),
        len(dataset) - int(frac * len(dataset))],
        generator=torch.Generator().manual_seed(0)
        )

    trainloader = DataLoader(trainset,shuffle=True,batch_size=batch_size)
    testloader = DataLoader(testset,shuffle=True,batch_size=batch_size)

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
    
    diffusion_config = torch.load(diffusion_weight_path)
    diffusion = Diffusion(
        noise_steps=diffusion_config['noise_steps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
        img_size=diffusion_config['img_size'],
        device=device
    )

    # 重建并加载 UNet 模型
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
    
    # 加载权重
    unet.load_state_dict(torch.load(unet_weight_path, map_location=device))
    unet.eval()

    # 加载权重
    unet.load_state_dict(torch.load(unet_weight_path, map_location=device))
    unet.eval()
    logger.success("✅ 模型加载完成（Step 2-2）")
    
    # # ---- 2-3 evaluation 1: loss ----
    # LOSS_SR = np.array([])
    # LOSS_BLU = np.array([])
    # valid_lossf = lossfunction.msejsloss
    # for batch_idx, (blurry_img, original_img) in enumerate(testloader):
    #     img_sr,jpt,jpt = model(blurry_img.detach())
    #     loss_sr = valid_lossf(img_sr,original_img).detach().cpu().numpy()
    #     loss_blurry = valid_lossf(blurry_img,original_img).detach().numpy()
    #     LOSS_SR = np.concat((LOSS_SR,loss_sr.flatten()))
    #     LOSS_BLU = np.concat((LOSS_BLU,loss_blurry.flatten()))
    # def hist(arr,color,nbins = 50,histtype = 'step',label = 'label'):
    #     #bins = np.logspace(np.log10(arr.min()),np.log10(arr.max()),nbins)
    #     #jpt = plt.hist(arr,bins = bins,density=True,histtype = histtype,color =color)
    #     #plt.xscale('log')
    #     bins = np.linspace((arr.min()),(arr.max()),nbins)
    #     jpt = plt.hist(arr,bins = bins,density=False,histtype = histtype,color =color,label = label)
    #     plt.legend()
    # plt.figure()
    # hist(LOSS_BLU,'red',label = 'blur')
    # hist(LOSS_SR,'blue',label = 'SR')
    # savepath = f'{ADDR_ROOT}/saves'
    # savefigname = f"Eval_loss_{model_name}_{exp_name}"
    # plt.savefig(f'{savepath}/EVAL/{savefigname}_jsdiv.png',dpi=300)
    # logger.info(f"Evaluation 1: Loss figure saved at {savepath}/EVAL/{savefigname}_jsdiv.png")
    # logger.success("✅ loss评估完成（Step 2-3-1）")
    
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
            position_encoding_dim=position_encoding_dim,
            position_encoding_function=positional_encoding,
            input_image=test_input_images[:n_images],
            save_time_steps=[0]
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
    savefigname = f"Eval_lineprofiles_{model_name}_{exp_name}"
    savefig_path = f'{savepath}/EVAL/{savefigname}.png'
    plt.savefig(savefig_path, dpi=300)
    logger.success(f"Evaluation 1: Loss figure saved at {savefig_path}")
    logger.success("✅ lineprofiles评估完成（Step 2-3-2）")
    
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
            position_encoding_dim=position_encoding_dim,
            position_encoding_function=positional_encoding,
            input_image=test_input_images[:n_images],
            save_time_steps=[0]
        )

    # 转置维度 (T, B, C, H, W) → (B, T, C, H, W)
    generated_images = generated_images.swapaxes(0, 1)

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
    
    plot_save_path = f"{savepath}/EVAL/evaluationplots_{model_name}_{exp_name}.png"
    plt.savefig(plot_save_path)
    plt.close()
    logger.info(f"Evaluation plots saved at {plot_save_path}")
    logger.success("✅ NRMSE,MAE,MS-SSIM,SSIM,PSNR评估完成（Step 2-3-2）")
    # -----------------------------------------

if __name__ == "__main__":
    app()
