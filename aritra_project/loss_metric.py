import torch
import math
import torch.nn as nn
import pytorch_ssim

criterion = nn.MSELoss()

def loss1(out_d, labels):
    l = (out_d - labels)
    loss_1 = 0.1 * torch.sum(torch.abs(l)) + 0.9 * torch.sqrt(torch.sum(l ** 2))
    return loss_1

def loss2(out_d, labels):
    l = (out_d - labels)
    loss_2 = torch.sqrt(torch.sum(l ** 2))
    return loss_2

def psnr(out_d, labels):
    mse = criterion(out_d, labels)
    metric = 10 * math.log10(1 / mse.item())
    return metric

def ssim(out_d, labels):
    metric1 = pytorch_ssim.ssim(out_d, labels)
    return metric1