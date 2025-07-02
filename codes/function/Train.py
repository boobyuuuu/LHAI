# ---- 1-2 Libraries for Configuration and Modules ----
from codes.config.config_cnn import TrainConfig
from codes.function.Dataset import ImageDataset
import codes.function.Loss as lossfunction
from codes.function.Log import log
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
import time
from datetime import timedelta

def train(model,optimizer,scheduler,trainingloss,device,dataloader,testloader,num_epochs,logger,logpath,train_msg="",LOSS_PLOT=[], TESTLOSS_PLOT=[], EPOCH_PLOT=[]):
    """
    函数特色：使用scheduler动态调整学习率。似乎没有其他在train函数进行优化的方式？
    """
    with open("training.log", "w", encoding="utf-8"):
        pass
    log(logpath,train_msg)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # 当前学习率
        current_lr = scheduler.get_last_lr()[0]

        for _, (img_LR, img_HR) in enumerate(dataloader):
            img_LR = img_LR.to(device)
            img_HR = img_HR.to(device)
            img_SR, _, _ = model(img_LR)
            loss = trainingloss(img_SR, img_HR)
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
            loss = trainingloss(img_SR, img_HR)
            test_loss += loss.item()

        test_avg_loss = test_loss / len(testloader)

        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, LR: {current_lr:.4e}")
        log(logpath,f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, LR: {current_lr:.4e}")

        LOSS_PLOT.append(avg_loss)
        TESTLOSS_PLOT.append(test_avg_loss)
        EPOCH_PLOT.append(epoch)

        # 更新学习率
        scheduler.step()


def train_diffusion(
    model,
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
        model: 待训练的模型
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
    with open("training_diffusion.log", "w", encoding="utf-8"):
        pass
    log(logpath,train_msg)
    for epoch in range(num_epochs):
        start_time = time.time()
        num_batches = len(trainloader)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}" + "\n" + "_" * 10)
        log(logpath,f"Epoch {epoch + 1}/{num_epochs}" + "\n" + "_" * 10)

        model.train()
        running_loss = 0.0

        current_lr = scheduler.get_last_lr()[0]

        for batch_idx, (input_images, target_images) in enumerate(trainloader):
            x_t, t, noise = prepare_data(input_images, target_images, diffusion, noise_steps, device, position_encoding_dim)

            outputs = model(x=x_t, t=t)
            optimizer.zero_grad()
            loss = criterion(outputs, noise)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                logger.info(f"Batch {batch_idx + 1}/{num_batches}: Train loss: {loss.item():.4f} LR: {current_lr:.6f}")
                log(logpath,f"Batch {batch_idx + 1}/{num_batches}: Train loss: {loss.item():.4f} LR: {current_lr:.6f}")

            running_loss += loss.item()

        train_loss.append(running_loss / num_batches)
        end_time = time.time()

        logger.info("-" * 10 + "\n" + f"Epoch {epoch + 1}/{num_epochs} : "
                    + f"Train loss: {train_loss[-1]:.4f}, "
                    + f"Time taken: {timedelta(seconds=end_time - start_time)}")
        log(logpath,"-" * 10 + "\n" + f"Epoch {epoch + 1}/{num_epochs} : "
                    + f"Train loss: {train_loss[-1]:.4f}, "
                    + f"Time taken: {timedelta(seconds=end_time - start_time)}")

        # 评估阶段
        model.eval()

        running_mae = 0.0
        running_ms_ssim = 0.0
        running_ssim = 0.0
        running_psnr = 0.0
        running_nrmse = 0.0

        n_images = 4

        with torch.no_grad():
            for batch_idx, (test_input_images, test_target_images) in enumerate(testloader):
                num_test_batches = len(testloader)

                test_input_images = test_input_images.to(device)
                test_target_images = test_target_images.to(device)

                generated_images = diffusion.reverse_diffusion(
                    model=model,
                    n_images=test_input_images.shape[0] if EVALUATE_METRICS else n_images,
                    n_channels=1,
                    position_encoding_dim=position_encoding_dim,
                    position_encoding_function=positional_encoding,
                    input_image=test_input_images if EVALUATE_METRICS else test_input_images[:n_images],
                    save_time_steps=[0],
                )

                if EVALUATE_METRICS:
                    generated_images_reshaped = generated_images.swapaxes(0, 1)[0]

                    batch_mae = mae_metric(generated_images_reshaped, test_target_images) if mae_metric else 0
                    batch_ms_ssim = ms_ssim_metric(generated_images_reshaped, test_target_images) if ms_ssim_metric else 0
                    batch_ssim = ssim_metric(generated_images_reshaped, test_target_images) if ssim_metric else 0
                    batch_psnr = psnr_metric(generated_images_reshaped, test_target_images) if psnr_metric else 0
                    batch_nrmse = torch.sqrt(
                        torch.mean((generated_images_reshaped - test_target_images) ** 2)
                    ) / (test_target_images.max() - test_target_images.min())

                    running_mae += batch_mae.item() if hasattr(batch_mae, 'item') else batch_mae
                    running_ms_ssim += batch_ms_ssim.item() if hasattr(batch_ms_ssim, 'item') else batch_ms_ssim
                    running_ssim += batch_ssim.item() if hasattr(batch_ssim, 'item') else batch_ssim
                    running_psnr += batch_psnr.item() if hasattr(batch_psnr, 'item') else batch_psnr
                    running_nrmse += batch_nrmse.item() if hasattr(batch_nrmse, 'item') else batch_nrmse

                else:
                    break

        if EVALUATE_METRICS and num_test_batches > 0:
            mae_results.append(running_mae / num_test_batches)
            ms_ssim_results.append(running_ms_ssim / num_test_batches)
            ssim_results.append(running_ssim / num_test_batches)
            psnr_results.append(running_psnr / num_test_batches)
            nrmse_results.append(running_nrmse / num_test_batches)

        # 更新学习率
        scheduler.step()
