# This py file function: evaluate the performance of the model

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
from codes.config.config_cnn import EvalConfig
from codes.function.Dataset import ImageDataset
from codes.function.Loss import jsdiv,jsdiv_single
from codes.function.Log import log
# ---- 1-3 Libraries for pytorch and others ----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from sklearn.metrics import mean_absolute_error as mae
import pytorch_msssim

app = typer.Typer()

# ---- 02 Define the main function ----
eval_cfg = EvalConfig()
@app.command()
def main(
    exp_name: str = eval_cfg.exp_name,                      # para01：实验名称 default: "EXP01"
    model_name: str = eval_cfg.model_name,                  # para02：模型名称 default: "CNN"
    model_dir: Path = eval_cfg.model_dir,                   # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_path: Path = eval_cfg.model_path,                 # para04：模型定义路径（.py） default: model_dir / f"{model_name}_{exp_name}.py"
    model_weight_dir: Path = eval_cfg.model_weight_dir,     # para05：模型权重目录 default: ADDR_ROOT / "saves" / "MODEL"
    model_weight_name: str = eval_cfg.model_weight_name,    # para06：模型权重文件名 default: "CNN_EXP01_400epo_32bth_xingwei.pth"
    model_weight_path: Path = eval_cfg.model_weight_path,   # para07：模型权重完整路径 default: model_weight_dir / model_weight_name
    data_dir: Path = eval_cfg.data_dir,                     # para08：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = eval_cfg.data_name,                    # para09：数据文件名 default: "xingwei_10000_64_train_v1.npy"
    data_path: Path = eval_cfg.data_path,                   # para10：数据完整路径 default: data_dir / data_name
    seed: int = eval_cfg.seed,                              # para11：随机种子 default: 0
    frac: float = eval_cfg.frac,                            # para12：训练集比例 default: 0.8
    batch_size: int = eval_cfg.batch_size,                  # para13：批次大小 default: 32
    epochs: int = eval_cfg.epochs,                          # para14：训练轮数（如需评估多个 epoch） default: 400
    lr_max: float = eval_cfg.lr_max,                        # para15：最大学习率 default: 5e-4
    lr_min: float = eval_cfg.lr_min                         # para16：最小学习率 default: 5e-6
):
    # ---- 2-1 Load the parameter ----
    logger.info("========== 当前训练参数 ==========")
    for idx, (key, value) in enumerate(locals().items(), start=1):
        logger.info(f"{idx:02d}. {key:<20}: {value}")
    torch.manual_seed(seed)
    spec = importlib.util.spec_from_file_location("module.name", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    MODEL = getattr(module, model_name)
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

    dataloader = DataLoader(trainset,shuffle=True,batch_size=batch_size)
    testloader = DataLoader(testset,shuffle=True,batch_size=batch_size)

    model = MODEL(0).to(device)
    logger.success("✅ 数据与模型加载完成（Step 2-2）")
    
    # ---- 2-3 evaluation 1: loss ----
    model.eval()
    model.cpu()
    LOSS_SR = np.array([])
    LOSS_BLU = np.array([])
    valid_lossf = jsdiv_single
    for batch_idx, (blurry_img, original_img) in enumerate(testloader):
        img_sr,jpt,jpt = model(blurry_img.detach())
        loss_sr = valid_lossf(img_sr,original_img).detach().cpu().numpy()
        loss_blurry = valid_lossf(blurry_img,original_img).detach().numpy()
        LOSS_SR = np.concat((LOSS_SR,loss_sr.flatten()))
        LOSS_BLU = np.concat((LOSS_BLU,loss_blurry.flatten()))
    def hist(arr,color,nbins = 50,histtype = 'step',label = 'label'):
        #bins = np.logspace(np.log10(arr.min()),np.log10(arr.max()),nbins)
        #jpt = plt.hist(arr,bins = bins,density=True,histtype = histtype,color =color)
        #plt.xscale('log')
        bins = np.linspace((arr.min()),(arr.max()),nbins)
        jpt = plt.hist(arr,bins = bins,density=False,histtype = histtype,color =color,label = label)
        plt.legend()
    plt.figure()
    hist(LOSS_BLU,'red',label = 'blur')
    hist(LOSS_SR,'blue',label = 'SR')
    savepath = f'{ADDR_ROOT}/saves'
    savefigname = f"Eval_loss_{model_name}_{exp_name}"
    plt.savefig(f'{savepath}/EVAL/{savefigname}_jsdiv.png',dpi=300)
    logger.success(f"Evaluation 1: Loss figure saved at {savepath}/EVAL/{savefigname}_jsdiv.png")
    
    # ---- 2-4 evaluation 2: distribution ----
    num_images_to_show = 10
    def interp2d(x1,x2,y1,y2,arr):
        x = np.arange(arr.shape[0])
        y = np.arange(arr.shape[0])
        xx,yy = np.meshgrid(x,y)
        interpolate = scipy.interpolate.RegularGridInterpolator((x,y),arr)
        y_t = np.linspace(x1,x2,101)
        x_t = np.linspace(y1,y2,101)
        z_t = interpolate((x_t,y_t))
        return z_t
    model.eval()
    model.to(device)
    img_LR=[] 
    img_HR=[]
    img_SR=[]

    showlist = [4,5]                    # ---- 2-4 更改1：选择要展示的图片编号 ----
    num_images_to_show = len(showlist)
    for i in showlist:
        item = trainset.__getitem__(i)
        img_LR.append(item[0])
        img_HR.append(item[1])
        img_SR.append(model((item[0].reshape(1,1,64,64).to(device)))[0].cpu())
    fig, axes = plt.subplots(num_images_to_show,6 , figsize=(18, 3 * num_images_to_show))
    xys = np.zeros(num_images_to_show).tolist()
    for i in range(num_images_to_show):
        xys[i] = [0,0,0,0]
    xys[0] = [30,50,20,10]              # ---- 2-4 更改2：调整图片的位置 ----
    xys[1]=[20,45,35,50]                # ---- 2-4 更改2：调整图片的位置 ----
    for i in range(num_images_to_show):
        color = 'white'
        x1,x2,y1,y2 = xys[i]
        blurry_img_numpy = img_LR[i].squeeze().detach().cpu().numpy()
        sr_img_numpy = img_SR[i].squeeze().detach().cpu().numpy()
        original_img_numpy = img_HR[i].squeeze().detach().cpu().numpy()
        blurry_img_numpy =blurry_img_numpy/blurry_img_numpy.sum()
        original_img_numpy=original_img_numpy/original_img_numpy.sum()
        sr_img_numpy =sr_img_numpy/sr_img_numpy.sum()
        im0=axes[i, 0].imshow(blurry_img_numpy)
        axes[i, 0].set_title('Blurry Image')
        axes[i,0].plot([x1,x2],[y1,y2],linestyle='--',color =color,linewidth = 2)
        axins = inset_axes(axes[i,0], width="20%", height="20%", loc=4)
        axins.axis('off')
        axins.patch.set_alpha(0)
        axins.plot(interp2d(x1,x2,y1,y2,blurry_img_numpy),color=color)
        axes[i,-1].plot(interp2d(x1,x2,y1,y2,blurry_img_numpy),color = 'red')
        im1=axes[i, 1].imshow(sr_img_numpy)
        axes[i, 1].set_title('SR Image')
        axes[i,1].plot([x1,x2],[y1,y2],linestyle='--',color = color,linewidth = 2)
        axins = inset_axes(axes[i,1], width="20%", height="20%", loc=4)
        axins.axis('off')
        axins.patch.set_alpha(0)
        axins.plot(interp2d(x1,x2,y1,y2,sr_img_numpy),color=color)
        axes[i,-1].plot(interp2d(x1,x2,y1,y2,sr_img_numpy),color = 'blue')
        im2=axes[i, 2].imshow(original_img_numpy)
        axes[i, 2].set_title('Original Image')
        axes[i,2].plot([x1,x2],[y1,y2],linestyle='--',color = color,linewidth = 2)
        axins = inset_axes(axes[i,2], width="20%", height="20%", loc=4)
        axins.axis('off')
        axins.patch.set_alpha(0)
        axins.plot(interp2d(x1,x2,y1,y2,original_img_numpy),color=color)
        axes[i,-1].plot(interp2d(x1,x2,y1,y2,original_img_numpy),color = 'black')
        res_blur = (blurry_img_numpy-original_img_numpy)
        res_sr = (sr_img_numpy-original_img_numpy)
        vmin = min(res_blur.min(),res_sr.min())
        vmax = max(res_blur.max(),res_sr.max())
        im3= axes[i, 3].imshow(res_blur,vmin =vmin,vmax=vmax)
        axes[i, 3].set_title('Res Blur')
        cbar2 = fig.colorbar(
            im3, ax=axes[i,3],shrink = 0.5
        )
        im4 = axes[i, 4].imshow(res_sr,vmin =vmin,vmax=vmax)
        axes[i, 4].set_title('Res SR')
        #axes[i, 4].axis('off')
        cbar2 = fig.colorbar(
            im4, ax=axes[i,4],shrink = 0.5
        )
    plt.tight_layout()
    savepath = f'{ADDR_ROOT}/saves'
    savefigname = f"Eval_distribution_{model_name}_{exp_name}"
    savefig_path = f'{savepath}/EVAL/{savefigname}.png'
    plt.savefig(savefig_path, dpi=300)
    logger.success(f"Evaluation 1: Loss figure saved at {savefig_path}")
    
    # ---- 2-5 evaluation 3: PSNR, SSIM ----
    psnr_list = []
    ssim_list = []
    mae_list = []
    mse_list = []
    nrmse_list = []
    ms_ssim_list = []
    
    output_path = f"{ADDR_ROOT}/saves/LOSS"
    output_file = f"{output_path}/Eval_PSNR_SSIM_{model_name}_{exp_name}.txt"
    
    model.eval()
    model.cpu()

    with open(output_file, "w") as f:
        f.write("Image_Index\tPSNR\tSSIM\tMS-SSIM\tMAE\tMSE\tNRMSE\n")  # 文件表头
        for idx, (blurry_img, original_img) in enumerate(testloader):
            # 生成超分辨图像
            with torch.no_grad():
                img_sr, _, _ = model(blurry_img)
            # 转为 numpy 格式
            blurry_img_np = blurry_img.squeeze().cpu().numpy()
            original_img_np = original_img.squeeze().cpu().numpy()
            img_sr_np = img_sr.squeeze().cpu().numpy()
            # 计算 PSNR 和 SSIM
            psnr_val = psnr(original_img_np, img_sr_np, data_range=original_img_np.max() - original_img_np.min())
            ssim_val = ssim(original_img_np, img_sr_np, data_range=original_img_np.max() - original_img_np.min())
            mae_val = mae(original_img_np, img_sr_np)
            mse_val = mse(original_img_np, img_sr_np)
            nrmse_val = nrmse(original_img_np, img_sr_np)
            # MS-SSIM (requires torch tensors, float32)
            img_sr_tensor = img_sr.float()
            original_tensor = original_img.float()
            ms_ssim_val = pytorch_msssim.ms_ssim(img_sr_tensor, original_tensor, data_range=1.0).item()
            # 保存结果到列表
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            ms_ssim_list.append(ms_ssim_val)
            mae_list.append(mae_val)
            mse_list.append(mse_val)
            nrmse_list.append(nrmse_val)
            # 写入 txt 文件
            f.write(f"{idx + 1}\t{psnr_val:.4f}\t{ssim_val:.4f}\t{ms_ssim_val:.4f}\t{mae_val:.4f}\t{mse_val:.4f}\t{nrmse_val:.4f}\n")
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_ms_ssim = np.mean(ms_ssim_list)
    avg_mae = np.mean(mae_list)
    avg_mse = np.mean(mse_list)
    avg_nrmse = np.mean(nrmse_list)

    # 打印日志
    logger.info(f"Average PSNR: {avg_psnr:.4f}")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Average MS-SSIM: {avg_ms_ssim:.4f}")
    logger.info(f"Average MAE: {avg_mae:.4f}")
    logger.info(f"Average MSE: {avg_mse:.4f}")
    logger.info(f"Average NRMSE: {avg_nrmse:.4f}")

    # 写入文件
    with open(output_file, "a") as f:
        f.write("\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average MS-SSIM: {avg_ms_ssim:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write(f"Average NRMSE: {avg_nrmse:.4f}\n")

    logger.success(f"All evaluation metrics saved at {output_file}")
    logger.success("Plot generation complete.")
    # -----------------------------------------

if __name__ == "__main__":
    app()
