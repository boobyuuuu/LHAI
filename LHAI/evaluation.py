# This py file function: evaluate the performance of the model

# ---- 01 Improt Libraries ----
# ---- 1-1 Libraries for Path and Logging ----
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import sys
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
PROJ_CODE = Path(__file__).resolve().parents[0]
logger.info(f"PROJ_CODE path is: {PROJ_CODE}")
# ---- 1-3 Libraries for Configuration ----
from config import EVAL_EXP_NAME, EVAL_MODEL_NAME, EVAL_MODEL_PYPATH, EVAL_MODEL_PTHNAME, EVAL_MODEL_PTHPATH, EVAL_DATA_DIR, EVAL_DATA_NAME, EVAL_DATA_PATH, EVAL_SEED, FRAC_TRAIN, BATCH_SIZE, LATENTDIM
from function.Dataset import ImageDataset
from function.Loss import jsdiv,jsdiv_single
from function.Log import log

# ---- 1-4 Libraries for pytorch and others ----
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

app = typer.Typer()

# ---- 02 Define the main function ----
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    EVAL_EXP_NAME: str = EVAL_EXP_NAME, # "EXP_0_1"
    EVAL_MODEL_NAME: str = EVAL_MODEL_NAME, # "CNN"
    EVAL_MODEL_PYPATH: Path = EVAL_MODEL_PYPATH, # "LHAI/LHAI/models/CNN_EXP_0_1.py"
    EVAL_MODEL_PTHNAME: str = EVAL_MODEL_PTHNAME, # "CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth"
    EVAL_MODEL_PTHPATH: Path = EVAL_MODEL_PTHPATH, # LHAI/saves/MODEL/CNN_EXP_0_1_400epo_32bth_64lat_poissonsrc+bkg_highresorig_poisson_src_bkg.pkl.npy.pth
    EVAL_DATA_DIR: Path = EVAL_DATA_DIR, # "LHAI/data/POISSON"
    EVAL_DATA_NAME: str = EVAL_DATA_NAME, # "poisson_src_bkg.pkl.npy"
    EVAL_DATA_PATH: Path = EVAL_DATA_PATH, # "LHAI/data/POISSON/poisson_src_bkg.pkl.npy"
    EVAL_SEED: int = EVAL_SEED, # 0
    frac_train: float = FRAC_TRAIN, # 0.8
    batch_size: int = BATCH_SIZE, # 32
    latentdim: int = LATENTDIM, # 64
    # -----------------------------------------
):
    # ---- 2-1 Load the parameter ----
    torch.manual_seed(EVAL_SEED)
    model_file_path = EVAL_MODEL_PYPATH
    spec = importlib.util.spec_from_file_location("module.name", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    MODEL = getattr(module, EVAL_MODEL_NAME)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"""
    Parameters:
    - EVAL_EXP_NAME: {EVAL_EXP_NAME}
    - EVAL_MODEL_NAME: {EVAL_MODEL_NAME}
    - EVAL_MODEL_PYPATH: {EVAL_MODEL_PYPATH}
    - EVAL_MODEL_PTHNAME: {EVAL_MODEL_PTHNAME}
    - EVAL_MODEL_PTHPATH: {EVAL_MODEL_PTHPATH}
    - EVAL_DATA_DIR: {EVAL_DATA_DIR}
    - EVAL_DATA_NAME: {EVAL_DATA_NAME}
    - EVAL_DATA_PATH: {EVAL_DATA_PATH}
    - EVAL_SEED: {EVAL_SEED}
    - FRAC_TRAIN: {frac_train}
    """)
    
    # ---- 2-2 Load the data and model ----
    filetmp = np.load(EVAL_DATA_PATH,allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    NUM_TO_LEARN = int(filelen)
    dataset = ImageDataset(NUM_TO_LEARN,EVAL_DATA_PATH,inverse=False)
    trainset, testset = random_split(dataset,
        lengths=[int(frac_train *len(dataset)),
        len(dataset) - int(frac_train * len(dataset))],
        generator=torch.Generator().manual_seed(0))
    dataloader = DataLoader(trainset,shuffle=True,batch_size=batch_size)
    testloader = DataLoader(testset,shuffle=True,batch_size=batch_size)
    if EVAL_MODEL_NAME == 'CNN':
        model = MODEL(0).to(device)
    elif EVAL_MODEL_NAME == 'VAE':
        model = MODEL(latentdim).to(device)
    elif EVAL_MODEL_NAME == 'GAN':
        model = MODEL().to(device)
    elif EVAL_MODEL_NAME == 'UNET':
        model = MODEL(0,0).to(device)
    else:
        model = MODEL(latentdim).to(device)
    logger.success(f"Data and Model.py loaded successfully.")
    lossfunction = jsdiv
    
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
    savepath = f'{PROJ_ROOT}/saves'
    savefigname = f"Eval_loss_{EVAL_MODEL_NAME}_{EVAL_EXP_NAME}"
    plt.savefig(f'{savepath}/FIGURE/{savefigname}_jsdiv.png',dpi=300)
    logger.success(f"Evaluation 1: Loss figure saved at {savepath}/FIGURE/{savefigname}_jsdiv.png")
    
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
    savepath = f'{PROJ_ROOT}/saves'
    savefigname = f"Eval_distribution_{EVAL_MODEL_NAME}_{EVAL_EXP_NAME}"
    savefig_path = f'{savepath}/FIGURE/{savefigname}.png'
    plt.savefig(savefig_path, dpi=300)
    logger.success(f"Evaluation 1: Loss figure saved at {savefig_path}")
    
    # ---- 2-5 evaluation 3: PSNR, SSIM ----
    psnr_list = []
    ssim_list = []
    output_path = f"{PROJ_ROOT}/saves/LOSS"
    output_file = f"{output_path}/Eval_PSNR_SSIM_{EVAL_MODEL_NAME}_{EVAL_EXP_NAME}.txt"
    model.eval()
    model.cpu()
    with open(output_file, "w") as f:
        f.write("Image_Index\tPSNR\tSSIM\n")  # 文件表头
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
            # 保存结果到列表
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            # 写入 txt 文件
            f.write(f"{idx + 1}\t{psnr_val:.4f}\t{ssim_val:.4f}\n")
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    logger.info(f"Average PSNR: {avg_psnr:.4f}")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    with open(output_file, "a") as f:
        f.write(f"\nAverage PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    logger.success(f"PSNR and SSIM results saved at {output_file}")
    # ---- 2 DONE ----
    logger.success("Plot generation complete.")
    # -----------------------------------------

if __name__ == "__main__":
    app()
