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
# ---- 1-2 Libraries for Configuration and Modules ----
from codes.config.config_cnn import TrainConfig
from codes.function.Dataset import ImageDataset
from codes.function.Loss import lossfunction
from codes.function.Log import log
# ---- 1-3 Libraries for pytorch and others ----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np

# ---- 02 Define the main function ----
train_cfg = TrainConfig()
app = typer.Typer()
@app.command()
def main(
    exp_name: str = train_cfg.exp_name,                  # para01：实验名称 default: "EXP01"
    model_name: str = train_cfg.model_name,              # para02：模型名称 default: "CNN"
    model_dir: Path = train_cfg.model_dir,               # para03：模型目录 default: ADDR_ROOT / "codes" / "models"
    model_path: Path = train_cfg.model_path,             # para04：模型路径 default: model_dir / f"{model_name}_{exp_name}.py"
    data_dir: Path = train_cfg.data_dir,                 # para05：数据目录 default: ADDR_ROOT / "data" / "Train"
    data_name: str = train_cfg.data_name,                # para06：数据文件名 default: "xingwei_10000_64_train_v1.npy"
    data_path: Path = train_cfg.data_path,               # para07：数据完整路径 default: data_dir / data_name
    seed: int = train_cfg.seed,                          # para08：随机种子 default: 0
    frac: float = train_cfg.frac,                        # para09：训练集比例 default: 0.8
    epochs: int = train_cfg.epochs,                      # para10：训练轮数 default: 400
    batch_size: int = train_cfg.batch_size,              # para11：批次大小 default: 32
    lr_max: float = train_cfg.lr_max,                    # para12：最大学习率 default: 5e-4
    lr_min: float = train_cfg.lr_min                     # para13：最小学习率 default: 5e-6
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
    LOSS_PLOT = []
    TESTLOSS_PLOT = []
    EPOCH_PLOT = []
    data_sim = data_name.split("_")[0]
    model_save_name = f"{model_name}_{exp_name}_{epochs}epo_{batch_size}bth_{data_sim}"
    logger.success("✅ 参数加载完成（Step 2-1）")
    
    # ---- 2-2 Load data ----
    filetmp = np.load(data_path, allow_pickle=True)
    filelen = filetmp.shape[0]
    del filetmp
    num_to_learn = int(filelen)

    dataset = ImageDataset(num_to_learn, data_path, inverse=False)
    trainset, testset = random_split(
        dataset,
        lengths=[int(frac * len(dataset)), len(dataset) - int(frac * len(dataset))],
        generator=torch.Generator().manual_seed(0)
    )

    dataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    logger.success("✅ 数据加载完成（Step 2-2）")

    # ---- 2-3 Initialize the model, loss function and optimizer ----
    model = MODEL(0).to(device)
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

    logger.success("✅ 模型加载完成（Step 2-3），准备开始训练")


    # ---- 2-4 Define the training function ----
    def train(dataloader, num_epochs):
        with open("training.log", "w"):
            pass
        log(f"Experiment name: {exp_name}")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / epochs))
            optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)

            for _, (img_LR, img_HR) in enumerate(dataloader):
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)
                img_SR, _, _ = model(img_LR)
                img_SR = img_SR.to(device)
                loss = lossfunction(img_SR, img_HR)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            test_loss = 0.0
            for _, (img_LR, img_HR) in enumerate(testloader):
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)
                img_SR, _, _ = model(img_LR)
                img_SR = img_SR.to(device)
                loss = lossfunction(img_SR, img_HR)
                test_loss += loss.item()

            test_avg_loss = test_loss / len(testloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, Current_LR:{current_lr:.4f}")
            log(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, Current_LR:{current_lr:.4f}")

            LOSS_PLOT.append(avg_loss)
            TESTLOSS_PLOT.append(test_avg_loss)
            EPOCH_PLOT.append(epoch)

    torch.set_printoptions(precision=10)

    logger.info(f"""
    Training Start...
    - DEVICE: {device}
    - Model: {model_name}
    - Experiment: {exp_name}
    """)
    train(dataloader, epochs)
    logger.success("✅ 模型训练完成（Step 2-4）")

    # ---- 2-5 Save the model and plot the loss ----
    fig, ax = plt.subplots()
    ax.plot(EPOCH_PLOT, LOSS_PLOT)
    ax.plot(EPOCH_PLOT, TESTLOSS_PLOT)
    ax.set_yscale('log')

    savepath = ADDR_ROOT / "saves"
    loss_plot_path = savepath / "LOSS" / f"{model_save_name}.png"
    loss_data_path = savepath / "LOSS" / f"{model_save_name}.npy"
    model_save_path = savepath / "MODEL" / f"{model_save_name}.pth"

    fig.savefig(loss_plot_path, dpi=300)
    logger.success(f"Loss plot saved at {loss_plot_path}")
    LOSS_DATA = np.stack((np.array(EPOCH_PLOT), np.array(LOSS_PLOT), np.array(TESTLOSS_PLOT)), axis=0)
    np.save(loss_data_path, LOSS_DATA)
    logger.success(f"Loss data saved at {loss_data_path}")
    torch.save(model.state_dict(), model_save_path)
    logger.success(f"Model saved at {model_save_path}")
    logger.success("✅ 模型保存完成（Step 2-5）")
    
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
    plt.savefig(f'{savepath}/PREDICT/EarlyPre_{model_save_name}.png', dpi = 300)
    logger.success(f"First prediction saved at {savepath}/PREDICT/FirstPred_{model_save_name}.png")
    logger.success("✅ 模型预测完成（Step 2-6）")
    # -----------------------------------------
if __name__ == "__main__":
    app()
