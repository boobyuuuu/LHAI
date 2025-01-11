# This py file function: Code to train models

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
from LHAI.config import EXP_NAME, MODEL_NAME, DATA_DIR, DATA_NAME, SEED, TRAINTYPE, FRAC_TRAIN, EPOCHS, BATCH_SIZE, LATENTDIM, LR_MAX, LR_MIN
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

# ---- 02 Define the main function ----
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    EXP_NAME: str = EXP_NAME, # para1: 实验名称
    MODEL_NAME: str = MODEL_NAME, # para2: 模型名称
    DATA_DIR: Path = DATA_DIR, # para3: 数据文件夹的路径
    DATA_NAME: str = DATA_NAME, # para4: 数据文件的名称
    seed: int = SEED, # para5: 随机种子
    traintype: str = TRAINTYPE, # para6: 训练类型
    frac_train: float = FRAC_TRAIN, # para7: 训练集比例
    epochs: int = EPOCHS, # para8: 训练轮数
    batch_size: int = BATCH_SIZE, # para9: 批次大小
    latentdim: int = LATENTDIM, # para10: 潜在维度
    lr_max: float = LR_MAX, # para11: 学习率上限
    lr_min: float = LR_MIN, # para12: 学习率下限
    MODEL_PATH: Path = PROJ_ROOT / "LHAI" / "models" / f"{MODEL_NAME}_{EXP_NAME}.py",
    DATA_PATH: Path = DATA_DIR / DATA_NAME,
    # ---- ADD ADDITIONAL ARGUMENTS AS NECESSARY ----
):
    
    # ---- 2-1 Load the parameter ----
    logger.info("Loading parameter, data and model...")
    torch.manual_seed(seed)
    MODEL_PATH = PROJ_ROOT / "LHAI" / "models" / f"{MODEL_NAME}_{EXP_NAME}.py"
    model_file_path = MODEL_PATH
    spec = importlib.util.spec_from_file_location("module.name", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    MODEL = getattr(module, MODEL_NAME)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LOSS_PLOT = []
    TESTLOSS_PLOT = []
    EPOCH_PLOT = []

    logger.info(f"""
    Parameters:
    - EXP_NAME: {EXP_NAME} (01)
    - MODEL_NAME: {MODEL_NAME} (02)
    - DATA_DIR: {DATA_DIR} (03)
    - DATA_NAME: {DATA_NAME} (04)
    - seed: {seed} (05)
    - traintype: {traintype} (06)
    - frac_train: {frac_train} (07)
    - epochs: {epochs} (08)
    - batch_size: {batch_size} (09)
    - latentdim: {latentdim} (10)
    - lr_max: {lr_max} (11)
    - lr_min: {lr_min} (12)
    - MODEL_PATH: {MODEL_PATH}
    - DATA_PATH: {DATA_PATH}
    """)
    
    MODEL_SAVE_NAME = f'{MODEL_NAME}_{EXP_NAME}_{EPOCHS}epo_{BATCH_SIZE}bth_{LATENTDIM}lat_{traintype}_{DATA_NAME}'
    
    # ---- 2-2 Load the data and model ----
    filetmp = np.load(DATA_PATH,allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    NUM_TO_LEARN = int(filelen)
    dataset = ImageDataset(NUM_TO_LEARN,DATA_PATH,inverse=False)
    trainset, testset = random_split(dataset,
        lengths=[int(frac_train *len(dataset)),
        len(dataset) - int(frac_train * len(dataset))],
        generator=torch.Generator().manual_seed(0))
    dataloader = DataLoader(trainset,shuffle=True,batch_size=batch_size)
    testloader = DataLoader(testset,shuffle=True,batch_size=batch_size)
    
    # ---- 2-3 Initialize the model, loss function and optimizer ----
    if MODEL_NAME == 'VAE':
        model = MODEL(latentdim).to(device)
    else:
        model = MODEL.to(device)
    lossfunction = lossfunction()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)

    logger.info("数据集的最大、最小值，以及一个样本")
    for batch_idx, (blurry_img, original_img) in enumerate(dataloader):
        if batch_idx == 0:
            blurry_img_shape = blurry_img.shape
            original_img_shape = original_img.shape
            blurry_img_numpy = blurry_img[1].squeeze().detach().numpy()
            blurry_img_min = blurry_img_numpy.min()
            blurry_img_max = blurry_img_numpy.max()
            blurry_img_sample = blurry_img_numpy
            break

    logger.info(f"""
    Dataset Sample Information:
    - Batch 1:
      - Blurry image shape: {blurry_img_shape}
      - Original image shape: {original_img_shape}
      - Blurry image min value: {blurry_img_min}
      - Blurry image max value: {blurry_img_max}
      - Blurry image sample: {blurry_img_sample}
    """)

    # ---- 2-4 Define the training function ----
    def train(dataloader, num_epochs):
        with open(f'training.log', 'w') as nothing: # 清空原log
            pass
        log(f"Experiment name: {EXP_NAME}")
        for epoch in range(num_epochs):
            model.train() # 切换成训练模式
            total_loss = 0.0
            current_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * epoch / EPOCHS))
            optimizer = torch.optim.AdamW(model.parameters(), lr = current_lr)

            for _, (img_LR, img_HR) in enumerate(dataloader):
                # img_LR = torch.squeeze(img_LR,dim = 1).to(DEVICE)
                # img_HR = torch.squeeze(img_HR,dim = 1).to(DEVICE)
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)
                img_SR, _, _ = model(img_LR)
                img_SR = img_SR.to(device)
                # 这步为止，img_LR,img_HR,img_SR均是[batchsize,不知道是什么,宽，高]
                #if epoch <= 500:
                #    loss = criterion1(img_SR, img_HR)
                #if epoch > 500:
                #    loss = criterion2(img_SR, img_HR) # 每个BATCH的loss，64张图平均
                loss = lossfunction(img_SR,img_HR)
                optimizer.zero_grad()
                loss.backward() # 最耗算力的一步
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader) # 每个EPOCH的loss，全部数据集的平均

            test_loss = 0.0
            for _, (img_LR, img_HR) in enumerate(testloader):
                # img_LR = torch.squeeze(img_LR,dim = 1).to(DEVICE)
                # img_HR = torch.squeeze(img_HR,dim = 1).to(DEVICE)
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)
                img_SR, _, _ = model(img_LR)
                img_SR = img_SR.to(device)
                # 这步为止，img_LR,img_HR,img_SR均是[batchsize,不知道是什么,宽，高]
                #if epoch <= 500:
                #    loss = criterion1(img_SR, img_HR)
                #if epoch > 500:
                #    loss = criterion2(img_SR, img_HR) # 每个BATCH的loss，64张图平均
                loss = lossfunction(img_SR,img_HR)
                test_loss += loss.item()
            test_avg_loss = test_loss / len(testloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, Current_LR:{current_lr:.4f}")
            log(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, Current_LR:{current_lr:.4f}")

            LOSS_PLOT.append(avg_loss)
            TESTLOSS_PLOT.append(test_avg_loss)
            EPOCH_PLOT.append(epoch)
            # if epoch % 300 == 0:
            #     torch.save(vae.state_dict(), Dir.TEMP()+'/checkpoint.pth')
            
    torch.set_printoptions(precision=10)

    logger.info(f"""
                Training Start...
                - DEVICE:{device}
                - Model:{MODEL_NAME}
                - Experiment Name:{EXP_NAME}
                """)
    train(dataloader, epochs)
    logger.success("Modeling training complete.")
    
    # ---- 2-5 Save the model and plot the loss ----
    fig,ax = plt.subplots()
    ax.plot(EPOCH_PLOT,LOSS_PLOT)
    ax.plot(EPOCH_PLOT,TESTLOSS_PLOT)
    ax.set_yscale('log')
    savepath = f'{PROJ_ROOT}/saves'
    fig.savefig(f'{savepath}/LOSS/{MODEL_SAVE_NAME}.png', dpi = 300)
    logger.info(f"Loss plot saved at {savepath}/LOSS/{MODEL_SAVE_NAME}.png")
    LOSS_DATA = np.stack((np.array(EPOCH_PLOT),np.array(LOSS_PLOT),np.array(TESTLOSS_PLOT)),axis=0)
    np.save(f'{savepath}/LOSS/{MODEL_SAVE_NAME}.npy',LOSS_DATA)
    logger.info(f"Loss data saved at {savepath}/LOSS/{MODEL_SAVE_NAME}.npy")
    torch.save(model.state_dict(), f'{savepath}/MODEL/{MODEL_SAVE_NAME}.pth')
    logger.info(f"Model saved at {savepath}/MODEL/{MODEL_SAVE_NAME}.pth")
    
    # ---- 2-6 early prediction ----
    model.eval()
    model.to(device)
    for _, (img_LR, img_HR) in enumerate(testloader):
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
    plt.savefig(f'{savepath}/FIGURE/EarlyPre_{MODEL_SAVE_NAME}.png', dpi = 300)
    logger.info(f"Early prediction saved at {savepath}/FIGURE/EarlyPre_{MODEL_SAVE_NAME}.png")
    # -----------------------------------------
if __name__ == "__main__":
    app()
