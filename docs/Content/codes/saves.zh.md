# LHAI项目代码：saves文件夹

`saves` 文件夹存放训练中产生的图像、损失数据；训练完成的模型；推理完成的结果图像


```
│── saves              <- 保存训练中的图像、损失数据、训练后的模型和预测结果
│   ├── FIGURE         <- 训练结束后初步推理形成的图像 
│                       (存储格式：./saves/FIGURE/EarlyPre_{MODEL_NAME}_{EXP_NAME}_{epochs}epo_{batch_size}bth_{latentdim}lat_{traintype}_{DATA_NAME}.png)
│   │── PRE_FIG        <- 推理时保存的图像
│                       (存储格式：./saves/PRE_FIG/{savepath}/PRE_FIG/Pre_{PRE_MODEL_NAME}.png)
│   ├── LOSS           <- 训练过程中保存的损失数据(.npy)和损失图像(.png) 
│                       (存储格式：./saves/LOSS/{MODEL_NAME}_{EXP_NAME}_{epochs}epo_{batch_size}bth_{latentdim}lat_{traintype}_{DATA_NAME}.npy + png)
│   └── MODEL          <- 保存的已训练模型和序列化模型
│                       （存储格式：./saves/MODEL/{MODEL_NAME}_{EXP_NAME}_{epochs}epo_{batch_size}bth_{latentdim}lat_{traintype}_{DATA_NAME}.pth）
```

这部分并没有什么好叙述的，训练/推理完成之后记得查看该文件夹获得结果。

<p align='right'>by Zihang Liu</p>