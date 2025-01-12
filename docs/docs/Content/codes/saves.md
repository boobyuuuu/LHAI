# LHAI Project Code: Saves Folder

The `saves` folder stores outputs generated during training, including images, loss data, trained models, and results from inference.

```
│── saves              <- Stores images, loss data, trained models, and prediction results
│   ├── FIGURE         <- Images generated from preliminary inference after training 
│   │                     (Format: ./saves/FIGURE/EarlyPre_{MODEL_NAME}_{EXP_NAME}_{epochs}epo_{batch_size}bth_{latentdim}lat_{traintype}_{DATA_NAME}.png)
│   │── PRE_FIG        <- Images saved during inference 
│   │                     (Format: ./saves/PRE_FIG/{savepath}/PRE_FIG/Pre_{PRE_MODEL_NAME}.png)
│   ├── LOSS           <- Loss data (.npy) and loss plots (.png) saved during training
│   │                     (Format: ./saves/LOSS/{MODEL_NAME}_{EXP_NAME}_{epochs}epo_{batch_size}bth_{latentdim}lat_{traintype}_{DATA_NAME}.npy + .png)
│   └── MODEL          <- Saved trained models and serialized files 
│                         (Format: ./saves/MODEL/{MODEL_NAME}_{EXP_NAME}_{epochs}epo_{batch_size}bth_{latentdim}lat_{traintype}_{DATA_NAME}.pth)
```

This folder requires minimal explanation -- once training or inference is complete, check this folder to retrieve the results.

<p align='right'>by Zihang Liu</p>