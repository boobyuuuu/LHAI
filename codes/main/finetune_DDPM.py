# DDPM fine-tuning entry point

import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

ADDR_ROOT = Path(__file__).resolve().parents[2]
ADDR_CODE = Path(__file__).resolve().parents[1]
sys.path.append(str(ADDR_ROOT))

from codes.config.config_DDPM import ModelConfig, TrainConfig
import codes.function.Loss as lossfunction
import codes.function.Train as Train

train_cfg = TrainConfig()
model_cfg = ModelConfig()
app = typer.Typer()


class ChannelPairDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, input_channel: int, target_channel: int, data_range: float = 1.0):
        data = np.load(data_path, allow_pickle=True)
        if data.ndim != 4:
            raise ValueError(f"Expected data shape (N,C,H,W), got {data.shape}")
        if not (0 <= input_channel < data.shape[1]) or not (0 <= target_channel < data.shape[1]):
            raise ValueError(f"Channel indices out of range for data shape {data.shape}")
        self.input_data = np.asarray(data[:, input_channel], dtype=np.float32)
        self.target_data = np.asarray(data[:, target_channel], dtype=np.float32)
        self.data_range = float(data_range)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.input_data.shape[0]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        denom = amax - amin
        if denom <= 0:
            return np.zeros_like(arr, dtype=np.float32)
        if abs(self.data_range - 1.0) < 1e-5:
            return ((arr - amin) / denom).astype(np.float32)
        if abs(self.data_range - 2.0) < 1e-5:
            return (2.0 * (arr - amin) / denom - 1.0).astype(np.float32)
        raise ValueError("datarange must be 1.0 or 2.0")

    def __getitem__(self, index: int):
        input_img = Image.fromarray(self._normalize(self.input_data[index]))
        target_img = Image.fromarray(self._normalize(self.target_data[index]))
        return self.transform(input_img), self.transform(target_img)


def build_channel_loaders(data_path: Path, input_channel: int, target_channel: int, batch_size: int, frac: float, data_range: float):
    dataset = ChannelPairDataset(data_path, input_channel=input_channel, target_channel=target_channel, data_range=data_range)
    train_size = int(float(frac) * len(dataset))
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(dataset)))
    trainloader = DataLoader(Subset(dataset, train_indices), shuffle=False, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    testloader = DataLoader(Subset(dataset, test_indices), shuffle=False, batch_size=batch_size, num_workers=0, pin_memory=False, drop_last=False)
    return trainloader, testloader


@app.command()
def main(
    pretrained_weight_path: Path = typer.Option(..., help="Existing DDPM/UNET .pth weight to fine-tune from."),
    output_name: str = typer.Option(..., help="Custom output name without .pth, e.g. FT_DDPM_KM2A_poissonexcess_EXP01."),
    exp_name: str = train_cfg.exp_name,
    data_dir: Path = train_cfg.data_dir,
    data_name: str = train_cfg.data_name,
    model_dir: Path = train_cfg.model_dir,
    model_name_diffusion: str = train_cfg.model_name_diffusion,
    model_name_unet: str = train_cfg.model_name_unet,
    seed: int = train_cfg.seed,
    frac: float = train_cfg.frac,
    epochs: int = typer.Option(20, help="Fine-tuning epochs."),
    batch_size: int = train_cfg.batch_size,
    lr_max: float = typer.Option(1e-5, help="Fine-tuning max learning rate."),
    lr_min: float = typer.Option(1e-6, help="Fine-tuning min learning rate."),
    datarange: float = train_cfg.datarange,
    input_channel: int = typer.Option(5, help="Input/blurry channel index in data array (N,C,H,W)."),
    target_channel: int = typer.Option(0, help="Target/GT channel index in data array (N,C,H,W)."),
    strict_load: bool = typer.Option(True, help="Use strict=True when loading pretrained weights."),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping max norm; <=0 disables clipping."),
):
    data_path = data_dir / data_name
    pretrained_weight_path = pretrained_weight_path.expanduser().resolve()
    if not pretrained_weight_path.exists():
        raise FileNotFoundError(pretrained_weight_path)

    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir_train = ADDR_ROOT / "saves" / "TRAIN" / model_name_diffusion
    save_dir_model = ADDR_ROOT / "saves" / "MODEL" / model_name_diffusion
    save_dir_train.mkdir(parents=True, exist_ok=True)
    save_dir_model.mkdir(parents=True, exist_ok=True)

    output_stem = output_name[:-4] if output_name.endswith(".pth") else output_name
    best_model_save_path = save_dir_model / f"Best_{output_stem}.pth"
    last_model_save_path = save_dir_model / f"Last_{output_stem}.pth"
    loss_plot_path = save_dir_train / f"{output_stem}.png"
    loss_data_path = save_dir_train / f"{output_stem}.npy"
    logpath = save_dir_train / f"finetune_log_{output_stem}.txt"

    trainloader, testloader = build_channel_loaders(
        data_path=data_path,
        input_channel=input_channel,
        target_channel=target_channel,
        batch_size=batch_size,
        frac=frac,
        data_range=datarange,
    )

    params_diffusion = dict(model_cfg.model_params[model_name_diffusion])
    params_diffusion["device"] = device
    params_unet = model_cfg.model_params[model_name_unet]

    sys.path.append(str(model_dir))
    DIFFUSION = getattr(importlib.import_module(model_name_diffusion), model_name_diffusion)
    UNET = getattr(importlib.import_module(model_name_unet), model_name_unet)

    diffusion = DIFFUSION(**params_diffusion)
    unet = UNET(**params_unet)
    unet.build()
    state_dict = torch.load(pretrained_weight_path, map_location=device)
    unet.load_state_dict(state_dict, strict=strict_load)
    unet = unet.to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr_max)
    lr_lambda = lambda epoch: lr_min / lr_max + 0.5 * (1 - lr_min / lr_max) * (1 + np.cos(np.pi * epoch / epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = lossfunction.msejsloss

    filetmp = np.load(data_path, allow_pickle=True)
    filelen = int(filetmp.shape[0])
    del filetmp

    train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
    train_msg = f"""
====================== DDPM Fine-tune 参数 ======================
traintime              : {train_time}
exp_name               : {exp_name}
pretrained_weight_path : {pretrained_weight_path}
output_name            : {output_stem}
model                  : {model_name_diffusion} + {model_name_unet}
data_path              : {data_path}
datalength             : {filelen}
input_channel          : {input_channel}
target_channel         : {target_channel}
frac                   : train {frac*100:.1f}% / test {100-frac*100:.1f}%
epochs                 : {epochs}
batch_size             : {batch_size}
datarange              : {datarange}
lr_max/lr_min          : {lr_max:.3e} / {lr_min:.3e}
strict_load            : {strict_load}
grad_clip              : {grad_clip}
device                 : {device} ({gpu_name})
best_model_save_path   : {best_model_save_path}
last_model_save_path   : {last_model_save_path}
=================================================================
"""
    logger.info(train_msg)

    loss_plot = []
    testloss_plot = []
    epoch_plot = []

    Train.train_DDPM(
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
        logpath=str(logpath),
        train_msg=train_msg,
        LOSS_PLOT=loss_plot,
        TESTLOSS_PLOT=testloss_plot,
        EPOCH_PLOT=epoch_plot,
        Best_model_save_path=str(best_model_save_path),
        grad_clip=grad_clip if grad_clip > 0 else None,
        eval_sample_every=None,
        save_time_steps=None,
    )

    torch.save(unet.state_dict(), last_model_save_path)
    logger.info(f"Last fine-tuned model saved at {last_model_save_path}")

    fig, ax = plt.subplots()
    ax.plot(epoch_plot, loss_plot, label="train")
    ax.plot(epoch_plot, testloss_plot, label="test")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(loss_plot_path, dpi=300)
    plt.close(fig)

    loss_data = np.stack((np.array(epoch_plot), np.array(loss_plot), np.array(testloss_plot)), axis=0)
    np.save(loss_data_path, loss_data)
    logger.info(f"Fine-tune loss plot saved at {loss_plot_path}")
    logger.info(f"Fine-tune loss data saved at {loss_data_path}")


if __name__ == "__main__":
    app()
