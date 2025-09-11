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

def train_DDPM(
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


def train_diffusion(
    unet,
    optimizer,
    scheduler,
    criterion,
    device,
    trainloader,
    testloader,
    diffusion,
    noise_steps,
    position_encoding_dim,
    positional_encoding,
    num_epochs,
    logger,
    logpath,
    train_msg,
    EVALUATE_METRICS=False,
    mae_metric=None,
    ms_ssim_metric=None,
    ssim_metric=None,
    psnr_metric=None,
    train_loss=None,
    mae_results=None,
    ms_ssim_results=None,
    ssim_results=None,
    psnr_results=None,
    nrmse_results=None,
):
    """
    训练函数，包含训练、测试和评估指标计算，支持学习率调度。

    参数:
        unet: 待训练的模型
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        device: 训练设备
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        diffusion: diffusion模型对象，含reverse_diffusion方法
        noise_steps: 噪声步数参数
        position_encoding_dim: 位置编码维度
        positional_encoding: 位置编码函数
        num_epochs: 训练轮数
        logger: 日志记录器
        logpath: 日志文件路径
        EVALUATE_METRICS: 是否计算评估指标，bool
        mae_metric, ms_ssim_metric, ssim_metric, psnr_metric: 评估指标函数（可选）
        train_loss, mae_results, ms_ssim_results, ssim_results, psnr_results, nrmse_results: 记录列表（外部传入以便保存历史）
    """
    with open(logpath, "w", encoding="utf-8"):
        pass
    log(logpath,train_msg)
    for epoch in range(num_epochs):
        start_time = time.time()
        num_batches = len(trainloader)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}" + "\n" + "_" * 10)
        log(logpath,f"Epoch {epoch + 1}/{num_epochs}" + "\n" + "_" * 10)

        unet.train()
        running_loss = 0.0

        current_lr = scheduler.get_last_lr()[0]

        for batch_idx, \
            (input_images, target_images) in enumerate(trainloader, start=0):
            x_t, t, noise = prepare_data(input_images, target_images, diffusion, noise_steps, device, position_encoding_dim)

            outputs = unet(x=x_t, t=t)
            optimizer.zero_grad()
            loss = criterion(outputs, noise)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}: " \
                    + f"Train loss: {loss.item():.4f}")
            running_loss += loss.item()

            train_loss.append(running_loss / num_batches)
            end_time = time.time()

        logger.info("-" * 10 + "\n" + f"Epoch {epoch + 1}/{num_epochs} : "
                    + f"Train loss: {train_loss[-1]:.4f}, "
                    + f"Time taken: {timedelta(seconds=end_time - start_time)}")
        log(logpath,"-" * 10 + "\n" + f"Epoch {epoch + 1}/{num_epochs} : "
                    + f"Train loss: {train_loss[-1]:.4f}, "
                    + f"Time taken: {timedelta(seconds=end_time - start_time)}")

        # # 评估阶段
        # unet.eval()

        # running_mae = 0.0
        # running_ms_ssim = 0.0
        # running_ssim = 0.0
        # running_psnr = 0.0
        # running_nrmse = 0.0

        # n_images = 4

        # for batch_idx, (test_input_images, \
        #             test_target_images) in enumerate(testloader):

        #     num_batches = len(testloader)
        #     test_input_images = test_input_images.to(device)
        #     test_target_images = test_target_images.to(device)

        #     generated_images = diffusion.reverse_diffusion(
        #         model=unet,
        #         n_images=test_input_images.shape[0] if EVALUATE_METRICS else n_images,
        #         n_channels=1,
        #         position_encoding_dim=position_encoding_dim,
        #         position_encoding_function=positional_encoding,
        #         input_image=test_input_images if EVALUATE_METRICS else \
        #             test_input_images[:n_images],
        #         save_time_steps=[0],
        #     )

        #     if EVALUATE_METRICS:
        #         generated_images_reshaped = generated_images.swapaxes(0, 1)[0]

        #         # Calculating the metrics for each batch
        #         batch_mae = mae_metric(generated_images_reshaped, test_target_images)
        #         batch_ms_ssim = ms_ssim_metric(generated_images_reshaped, test_target_images)
        #         batch_ssim = ssim_metric(generated_images_reshaped, test_target_images)
        #         batch_psnr = psnr_metric(generated_images_reshaped, test_target_images)
        #         batch_nrmse = torch.sqrt(
        #             torch.mean((generated_images_reshaped - test_target_images) ** 2)
        #         ) / (test_target_images.max() - test_target_images.min())

        #         # Accumulating the metrics
        #         running_mae += batch_mae.item()
        #         running_ms_ssim += batch_ms_ssim.item()
        #         running_ssim += batch_ssim.item()
        #         running_psnr += batch_psnr.item()
        #         running_nrmse += batch_nrmse.item()

        #     else:
        #         break

        # if EVALUATE_METRICS:
        #     # Storing the mean metric values per epoch to the empty lists
        #     mae_results.append(running_mae / num_batches)
        #     ms_ssim_results.append(running_ms_ssim / num_batches)
        #     ssim_results.append(running_ssim / num_batches)
        #     psnr_results.append(running_psnr / num_batches)
        #     nrmse_results.append(running_nrmse / num_batches)

        # 更新学习率
        scheduler.step()
