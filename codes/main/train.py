# This py file function: Code to train models

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
# ---- 1-3 Libraries for Configuration and Modules ----
import codes.config.config_cnn as config
from codes.function.Dataset import ImageDataset
from codes.function.Loss import lossfunction
from codes.function.Log import log
# ---- 1-4 Libraries for pytorch and others ----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np

# ---- 02 Define the main function ----
app = typer.Typer()
@app.command()
def main(
    TRAIN_EXP_NAME: str = config.TRAIN_EXP_NAME,        # para1：实验名称 default："EXP01"
    TRAIN_MODEL_NAME: str = config.TRAIN_MODEL_NAME,    # para2：模型名称 default："CNN"
    TRAIN_MODEL_PY: Path = config.TRAIN_MODEL_PY,       # para13：模型.py文件路径 default：ADDR_ROOT / "codes" / "models" / f"{TRAIN_MODEL_NAME}_{TRAIN_EXP_NAME}.py"
    TRAIN_DATA_DIR: Path = config.TRAIN_DATA_DIR,       # para3：数据文件夹路径 default：ADDR_ROOT / "data" / "POISSON"
    TRAIN_DATA_NAME: str = config.TRAIN_DATA_NAME,      # para4：数据文件名 default："poisson_src_bkg.pkl.npy"
    TRAIN_DATA_PATH: Path = config.TRAIN_DATA_PATH,      # para14：完整数据路径 default：TRAIN_DATA_DIR / TRAIN_DATA_NAME
    TRAIN_SEED: int = config.TRAIN_SEED,                # para5：随机种子 default：0
    TRAIN_FRAC: float = config.TRAIN_FRAC,              # para7：训练集比例 default：0.8
    TRAIN_EPOCHS: int = config.TRAIN_EPOCHS,            # para8：训练轮数 default：400
    TRAIN_BATCH_SIZE: int = config.TRAIN_BATCH_SIZE,    # para9：批次大小 default：32
    TRAIN_LR_MAX: float = config.TRAIN_LR_MAX,          # para11：学习率上限 default：5e-4
    TRAIN_LR_MIN: float = config.TRAIN_LR_MIN          # para12：学习率下限 default：5e-6
):
    
    # ---- 2-1 Load the parameter ----
    logger.info("========== 当前训练参数 ==========")
    for idx, (key, value) in enumerate(locals().items(), start=1):
        logger.info(f"{idx:02d}. {key:<20}: {value}")
    torch.manual_seed(TRAIN_SEED)
    model_file_path = TRAIN_MODEL_PY
    spec = importlib.util.spec_from_file_location("module.name", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    MODEL = getattr(module, TRAIN_MODEL_NAME)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LOSS_PLOT = []
    TESTLOSS_PLOT = []
    EPOCH_PLOT = []
    DATA_SIM = TRAIN_DATA_NAME.split("_")[0]
    
    MODEL_SAVE_NAME = f'{TRAIN_MODEL_NAME}_{TRAIN_EXP_NAME}_{TRAIN_EPOCHS}epo_{TRAIN_BATCH_SIZE}bth_{DATA_SIM}'
    # print(MODEL_SAVE_NAME)
    logger.success("2-1 Loading parameters")
    
    # ---- 2-2 Load data----
    filetmp = np.load(TRAIN_DATA_PATH,allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    NUM_TO_LEARN = int(filelen)
    dataset = ImageDataset(NUM_TO_LEARN,TRAIN_DATA_PATH,inverse=False)
    trainset, testset = random_split(dataset,
        lengths=[int(TRAIN_FRAC *len(dataset)),
        len(dataset) - int(TRAIN_FRAC * len(dataset))],
        generator=torch.Generator().manual_seed(0))
    dataloader = DataLoader(trainset,shuffle=True,batch_size=TRAIN_BATCH_SIZE)
    testloader = DataLoader(testset,shuffle=True,batch_size=TRAIN_BATCH_SIZE)
    logger.success("2-2 Loading data")
    
    # ---- 2-3 Initialize the model, loss function and optimizer ----
    model = MODEL(0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR_MAX)

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
    logger.success("2-3 Loading model")

    # ---- 2-4 Define the training function ----
    def train(dataloader, num_epochs):
        with open(f'training.log', 'w') as nothing: # 清空原log
            pass
        log(f"Experiment name: {TRAIN_EXP_NAME}")
        for epoch in range(num_epochs):
            model.train() # 切换成训练模式
            total_loss = 0.0
            current_lr = TRAIN_LR_MIN + 0.5 * (TRAIN_LR_MAX - TRAIN_LR_MIN) * (1 + np.cos(np.pi * epoch / TRAIN_EPOCHS))
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
                - Model:{TRAIN_MODEL_NAME}
                - Experiment:{TRAIN_EXP_NAME}
                """)
    train(dataloader, TRAIN_EPOCHS)
    logger.success("Modeling training complete.")
    
    # ---- 2-5 Save the model and plot the loss ----
    fig,ax = plt.subplots()
    ax.plot(EPOCH_PLOT,LOSS_PLOT)
    ax.plot(EPOCH_PLOT,TESTLOSS_PLOT)
    ax.set_yscale('log')
    savepath = f'{ADDR_ROOT}/saves'
    fig.savefig(f'{savepath}/LOSS/{MODEL_SAVE_NAME}.png', dpi = 300)
    logger.success(f"Loss plot saved at {savepath}/LOSS/{MODEL_SAVE_NAME}.png")
    LOSS_DATA = np.stack((np.array(EPOCH_PLOT),np.array(LOSS_PLOT),np.array(TESTLOSS_PLOT)),axis=0)
    np.save(f'{savepath}/LOSS/{MODEL_SAVE_NAME}.npy',LOSS_DATA)
    logger.success(f"Loss data saved at {savepath}/LOSS/{MODEL_SAVE_NAME}.npy")
    torch.save(model.state_dict(), f'{savepath}/MODEL/{MODEL_SAVE_NAME}.pth')
    logger.success(f"Model saved at {savepath}/MODEL/{MODEL_SAVE_NAME}.pth")
    
    # ---- 2-6 First prediction ----
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
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{savepath}/PREDICT/EarlyPre_{MODEL_SAVE_NAME}.png', dpi = 300)
    logger.success(f"First prediction saved at {savepath}/PREDICT/FirstPred_{MODEL_SAVE_NAME}.png")
    # -----------------------------------------
if __name__ == "__main__":
    app()
