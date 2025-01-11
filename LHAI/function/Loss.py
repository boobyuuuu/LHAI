import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

# ---- 01 ssim ----
def ssim_function(img1, img2, window_size=11, data_range=255.0, sigma=1.5):
    K1 = 0.01
    K2 = 0.03

    # 将图像转换为numpy数组
    #img1 = np.array(img1.astype(np.float32)
    #img2 = np.array(img2).astype(np.float32)

    # 计算SSIM的常数
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 高斯滤波
    window = np.outer(gaussian(window_size, sigma), gaussian(window_size, sigma))

    # 计算均值
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')

    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    # 计算SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    # 返回平均SSIM
    return np.mean(ssim_map)

def gaussian(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()


# ---- 02 mse ----
# 结构相似度
def batch_ssim(img1, img2):
    ssim = torch.zeros(img1.size(0))
    for i in range(img1.size(0)):
        img1_pil = to_pil_image(img1[i])
        img2_pil = to_pil_image(img2[i])
        ssim[i] = ssim_function(img1_pil, img2_pil, window_size = 16, data_range = 255.0, sigma = 1.5)
    return ssim.mean()

# 峰值信噪比
def batch_psnr(img1, img2, max_val=255.0):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)).cpu().detach().numpy()
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse)))
    return torch.mean(psnr)

class Custom_criterion(nn.Module):
    def __init__(self):
        super(Custom_criterion, self).__init__()
        self.mse_weight = 0.6
        self.ssim_weight = 0.4
        self.psnr_weight = 0
        self.l1_weight = 0

    def forward(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        ssim_loss = 1 - batch_ssim(output, target) # 取1-，因为越接近1越好
        #psnr_loss = -batch_psnr(output, target)  # 取相反数，因为 PSNR 越大越好
        #l1_loss = nn.L1Loss()(output, target)
        mse = mse_loss * self.mse_weight
        ssim = ssim_loss * self.ssim_weight
        #psnr = psnr_loss * self.psnr_weight
        #l1 = l1_loss * self.l1_weight

        return ssim_loss
        #return mse + ssim
        #return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        #return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss + self.psnr_weight * psnr_loss

# ---- 03 jsdiv ----
def jsdiv(img1, img2):
    shape = img2.shape
    img1 = img1.reshape(shape[0],shape[1],-1)
    img2 = img2.reshape(shape[0],shape[1],-1)
    img1 = F.softmax(img1,-1)
    img2 = F.softmax(img2,-1)
    img1 = img1.reshape(shape)
    img2 = img2.reshape(shape)
    img_mean = (img1/2+img2/2)
    ks12 = (img1*torch.log(img1/img_mean)).sum(-1).sum(-1)
    ks21 = (img2*torch.log(img2/img_mean)).sum(-1).sum(-1)
    #print((img1*torch.log(img1/img2)))
    
    jsdivergence = ((ks12+ks21)/2).mean()
    #print(jsdivergence)
    #raise ValueError('Stop')
    
    return jsdivergence

def jsdiv_single(img1, img2):
    shape = img2.shape
    img1 = img1.reshape(shape[0],shape[1],-1)
    img2 = img2.reshape(shape[0],shape[1],-1)
    img1 = F.softmax(img1,-1)
    img2 = F.softmax(img2,-1)
    img1 = img1.reshape(shape)
    img2 = img2.reshape(shape)
    img_mean = (img1/2+img2/2)
    ks12 = (img1*torch.log(img1/img_mean)).sum(-1).sum(-1)
    ks21 = (img2*torch.log(img2/img_mean)).sum(-1).sum(-1)
    #print((img1*torch.log(img1/img2)))
    
    jsdivergence = ((ks12+ks21)/2)#.mean()
    #print(jsdivergence)
    #raise ValueError('Stop')
    
    return jsdivergence

# 使用示例
# img1和img2需要是torch张量，并且在CUDA设备上
# loss = ssim_function(img1, img2)

# ---- 04 kl ----
# ---- 05 l1 ----
# ---- 06 l2 ----
# ---- 07 mae ----
# ---- 08 rmse ----
# ---- 09 psnr ----
# ---- 10 vgg ----
# ---- 11 cosine ----
# ---- 12 hinge ----
# ---- 13 triplet ----
# ---- 14 contrastive ----
# ---- 15 nll ----
# ---- 16 bce ----
# ---- 17 dice ----
# ---- 18 focal ----
# ---- 19 tversky ----
# ---- 20 gdl ----
# ---- 21 hinge ----

# ---- 22 custom ----

def lossfunction(img1,img2):
    mse = nn.MSELoss()
    loss = mse(img1,img2)*0.2 + jsdiv(img1,img2)*0.8
    return loss