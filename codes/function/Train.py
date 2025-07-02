# ---- 1-2 Libraries for Configuration and Modules ----
from codes.config.config_cnn import TrainConfig
from codes.function.Dataset import ImageDataset
import codes.function.Loss as lossfunction
from codes.function.Log import log
# ---- 1-3 Libraries for pytorch and others ----
import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import DataLoader,random_split, ConcatDataset
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
import numpy as np

def train(model,optimizer,scheduler,trainingloss,device,dataloader,testloader,num_epochs,logger,train_msg="",LOSS_PLOT=[], TESTLOSS_PLOT=[], EPOCH_PLOT=[]):
        with open("training.log", "w", encoding="utf-8"):
            pass
        log(train_msg)
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
            log(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4e}, Test Loss: {test_avg_loss:.4e}, LR: {current_lr:.4e}")

            LOSS_PLOT.append(avg_loss)
            TESTLOSS_PLOT.append(test_avg_loss)
            EPOCH_PLOT.append(epoch)

            # 更新学习率
            scheduler.step()