# ---- 1-2 Libraries for Configuration and Modules ----
from codes.config.config_cnn import TrainConfig
from codes.function.Dataset import ImageDataset
import codes.function.Loss as lossfunction
from codes.function.Log import log
from codes.models.DIFFUSION import positional_encoding
from codes.models.DIFFUSION import prepare_data
# ---- 1-3 Libraries for pytorch and others ----
import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import os
from typing import List, Optional
from torch.nn.utils import clip_grad_norm_

def format_model_params(params: dict, indent: int = 8) -> str:
    max_len = max(len(str(k)) for k in params)
    pad = " " * indent  # 控制缩进宽度
    return "\n".join(
        f"{pad}• {k:<{max_len}} : {v}"
        for k, v in params.items()
    )

def train_cnn(
    model,
    optimizer,
    scheduler,
    criterion,
    device,
    trainloader,
    testloader,
    num_epochs,
    logger,
    logpath,
    train_msg="",
    LOSS_PLOT=[],
    TESTLOSS_PLOT=[],
    EPOCH_PLOT=[],
    Best_model_save_path=""
):
    """
    函数特色：使用scheduler动态调整学习率。似乎没有其他在train函数进行优化的方式？
    """
    with open(logpath, "w", encoding="utf-8"):
        pass
    log(logpath,train_msg)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        for _, (img_LR, img_HR) in enumerate(trainloader):
            img_LR = img_LR.to(device)
            img_HR = img_HR.to(device)
            img_SR, _, _ = model(img_LR)
            loss = criterion(img_SR, img_HR)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)

        # ===== 验证阶段 =====
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for _, (img_LR, img_HR) in enumerate(testloader):
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)
                img_SR, _, _ = model(img_LR)
                loss = criterion(img_SR, img_HR)
                test_loss += loss.item()
        test_avg_loss = test_loss / len(testloader)

        # ===== 日志输出 =====
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, LR: {current_lr:.4e}")
        log(logpath,f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, LR: {current_lr:.4e}")

        LOSS_PLOT.append(avg_loss)
        TESTLOSS_PLOT.append(test_avg_loss)
        EPOCH_PLOT.append(epoch)

        # ===== 保存最佳模型 =====
        if test_avg_loss < best_loss:
            best_loss = test_avg_loss
            torch.save(model.state_dict(), Best_model_save_path)
            logger.success(f"💾 发现更优模型（Test Loss={best_loss:.4e}），已保存：{Best_model_save_path} (Epoch {epoch+1})")

        scheduler.step()

def train_diffusion(
    unet,                       # 噪声预测网络（如 AttentionUNet 包装）
    diffusion,                  # 你的 DDPM_Transformer / DDPM 实例
    optimizer,
    scheduler,
    criterion,                  # 一般用 nn.MSELoss()，比较 predicted_noise vs true_noise
    device,
    trainloader,
    testloader,
    num_epochs: int,
    logger,
    logpath: str,
    train_msg: str = "",
    LOSS_PLOT: Optional[List[float]] = None,
    TESTLOSS_PLOT: Optional[List[float]] = None,
    EPOCH_PLOT: Optional[List[int]] = None,
    Best_model_save_path: str = "",
    grad_clip: Optional[float] = None,  # e.g., 1.0；None 表示不裁剪
    eval_sample_every: Optional[int] = None,  # e.g., 10；None 表示不做采样评估
    save_time_steps: Optional[List[int]] = None,  # 需要可视化中间过程时传入
):
    """
    训练 DDPM（噪声预测）版的超分辨：
    - 训练目标：min E[ || eps_pred - eps_true ||^2 ]
    - 验证同训练（不走完整反向采样，速度快且稳定）
    - 可选：每 N 个 epoch 进行一次小样本反推采样（耗时，默认关闭）

    说明：
    - diffusion.sample_training_batch 会完成：
        1) 采样 t
        2) 前向加噪得到 x_t 与真实噪声 noise
        3) 若 conditional=True，则将 input_img 与 x_t 在通道维拼接
        4) 返回 (x_t_concat, t_emb, noise)
    """

    # --- 初始化绘图缓存 ---
    LOSS_PLOT = [] if LOSS_PLOT is None else LOSS_PLOT
    TESTLOSS_PLOT = [] if TESTLOSS_PLOT is None else TESTLOSS_PLOT
    EPOCH_PLOT = [] if EPOCH_PLOT is None else EPOCH_PLOT

    # --- 重置日志文件 ---
    with open(logpath, "w", encoding="utf-8"):
        pass
    log(logpath, train_msg)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        unet.train()
        total_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        # ====== 训练 ======
        for _, (img_LR, img_HR) in enumerate(trainloader):
            img_LR = img_LR.to(device)
            img_HR = img_HR.to(device)

            # 准备带噪输入与时间步嵌入
            x_t, t_emb, noise_true = diffusion.sample_training_batch(
                input_img=img_LR, target_img=img_HR
            )

            # 预测噪声
            noise_pred = unet(x_t, t_emb)

            loss = criterion(noise_pred, noise_true)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(unet.parameters(), max_norm=grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(trainloader))

        # ====== 验证 ======
        test_loss = 0.0
        unet.eval()
        with torch.no_grad():
            for _, (img_LR, img_HR) in enumerate(testloader):
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)

                x_t, t_emb, noise_true = diffusion.sample_training_batch(
                    input_img=img_LR, target_img=img_HR
                )
                noise_pred = unet(x_t, t_emb)
                loss = criterion(noise_pred, noise_true)
                test_loss += loss.item()

        test_avg_loss = test_loss / max(1, len(testloader))

        # ====== 日志 ======
        logger.info(
            f"[DDPM] Epoch [{epoch+1}/{num_epochs}] "
            f"Train: {avg_loss:.4e} | Val: {test_avg_loss:.4e} | LR: {current_lr:.4e}"
        )
        log(
            logpath,
            f"[DDPM] Epoch [{epoch+1}/{num_epochs}] "
            f"Train: {avg_loss:.4e} | Val: {test_avg_loss:.4e} | LR: {current_lr:.4e}"
        )

        LOSS_PLOT.append(avg_loss)
        TESTLOSS_PLOT.append(test_avg_loss)
        EPOCH_PLOT.append(epoch)

        # ====== 保存最优权重（按验证集噪声预测损失） ======
        if test_avg_loss < best_loss and Best_model_save_path:
            best_loss = test_avg_loss
            torch.save(unet.state_dict(), Best_model_save_path)
            logger.success(
                f"💾 发现更优模型（Val NoiseLoss={best_loss:.4e}），已保存：{Best_model_save_path} (Epoch {epoch+1})"
            )

        # ====== 可选：小样本反向采样（慢，默认关闭） ======
        do_sample = (eval_sample_every is not None) and ((epoch + 1) % eval_sample_every == 0)
        if do_sample:
            try:
                # 取一个小 batch 做示例（比如前 4 张）
                img_LR, img_HR = next(iter(testloader))
                img_LR = img_LR.to(device)
                img_HR = img_HR.to(device)
                n = min(4, img_LR.size(0))
                # 反向扩散（条件采样）
                samples = diffusion.reverse_diffusion(
                    model=unet,
                    n_images=n,
                    n_channels=img_HR.size(1),
                    input_image=img_LR[:n] if getattr(diffusion, "conditional", True) else None,
                    save_time_steps=save_time_steps
                )
                # 这里不强制计算 PSNR/SSIM（因数据归一化不同），如需可在此处添加
                logger.info(f"[DDPM] Epoch {epoch+1}: 采样完成（示例 n={n}）")
            except Exception as e:
                logger.warning(f"[DDPM] 采样失败（跳过）：{e}")

        scheduler.step()