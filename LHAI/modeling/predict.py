# This py file function: Code to evaluate models simply

# ---- 01 Improt Libraries ----
# ---- 1-1 Libraries for Path and Logging ----
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import sys
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
PROJ_CODE = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_ROOT))
logger.info(f"PROJ_CODE path is: {PROJ_CODE}")
# ---- 1-3 Libraries for Configuration ----
from LHAI.config import PRE_MODEL_PATH, PRE_DATA_PATH, PRE_MODEL_NAME, PRE_MODEL, PRE_DATA_NAME, PRE_SEED, PRE_TRAINTYPE, PRE_FRAC_TRAIN, PRE_BATCH_SIZE, PRE_LATENT_DIM
from LHAI.function.Dataset import ImageDataset
from LHAI.function.Loss import lossfunction
from LHAI.function.Log import log

# ---- 1-4 Libraries for pytorch and others ----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np

app = typer.Typer()


@app.command()
def main(
    PRE_MODEL_PATH: Path = PRE_MODEL_PATH,
    PRE_DATA_PATH: Path = PRE_DATA_PATH,
    PRE_MODEL_NAME: str = PRE_MODEL_NAME,
    PRE_MODEL: str = PRE_MODEL,
    PRE_DATA_NAME: str = PRE_DATA_NAME,
    PRE_SEED: int = PRE_SEED,
    PRE_TRAINTYPE: str = PRE_TRAINTYPE,
    PRE_FRAC_TRAIN: float = PRE_FRAC_TRAIN,
    PRE_BATCH_SIZE: int = PRE_BATCH_SIZE,
    PRE_LATENT_DIM: int = PRE_LATENT_DIM,
    # -----------------------------------------
):
    
    torch.manual_seed(PRE_SEED)
    MODEL_PATH = PRE_MODEL_PATH / PRE_MODEL_NAME
    DATA_PATH = PRE_DATA_PATH / PRE_DATA_NAME
    model_file_path = MODEL_PATH
    spec = importlib.util.spec_from_file_location("module.name", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    MODEL = getattr(module, PRE_MODEL)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"""
    Parameters:
    - PRE_MODEL_PATH: {PRE_MODEL_PATH} (01)
    - PRE_DATA_PATH: {PRE_DATA_PATH} (02)
    - PRE_MODEL_NAME: {PRE_MODEL_NAME} (03)
    - PRE_MODEL: {PRE_MODEL} (04)
    - PRE_DATA_NAME: {PRE_DATA_NAME} (05)
    - PRE_SEED: {PRE_SEED} (06)
    - PRE_TRAINTYPE: {PRE_TRAINTYPE} (07)
    - PRE_FRAC_TRAIN: {PRE_FRAC_TRAIN} (08)
    """)
    filetmp = np.load(DATA_PATH,allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    NUM_TO_LEARN = int(filelen)
    NUM_TO_PRE = int(filelen*(1-PRE_FRAC_TRAIN))
    dataset = ImageDataset(NUM_TO_PRE,DATA_PATH,inverse=True)
    dataloader = DataLoader(dataset, batch_size=PRE_BATCH_SIZE, shuffle=True)
    if PRE_MODEL == 'CNN':
        model = MODEL(0).to(device)
    elif PRE_MODEL == 'VAE':
        model = MODEL(PRE_LATENT_DIM).to(device)
    elif PRE_MODEL == 'GAN':
        model = MODEL().to(device)
    elif PRE_MODEL == 'UNET':
        model = MODEL(0,0).to(device)
    else:
        model = MODEL(PRE_LATENT_DIM).to(device)
    # ---- START PREDICTION ----
    savepath = f'{PROJ_ROOT}/saves'
    MODEL_SAVE_NAME = PRE_MODEL_NAME
    model.eval()
    model.to(device)
    for _, (img_LR, img_HR) in enumerate(dataloader):
        #print(img_LR.shape)
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

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{savepath}/PRE_FIG/Pre_{MODEL_SAVE_NAME}.png', dpi = 300)
    logger.info(f"Early prediction saved at {savepath}/PRE_FIG/Pre_{MODEL_SAVE_NAME}.png")
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.success("Prediction complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()