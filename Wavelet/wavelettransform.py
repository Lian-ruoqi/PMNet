import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()
    
    def dwt(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B*C, 1, H, W)

        x1 = x[:, :, 0::2, 0::2] / np.sqrt(2)
        x2 = x[:, :, 0::2, 1::2] / np.sqrt(2)
        x3 = x[:, :, 1::2, 0::2] / np.sqrt(2)
        x4 = x[:, :, 1::2, 1::2] / np.sqrt(2)
        
        LL = x1 + x2 + x3 + x4
        LH = -x1 - x3 + x2 + x4
        HL = -x1 + x3 - x2 + x4
        HH = x1 - x3 - x2 + x4

        return torch.cat([LL, LH, HL, HH], dim=1).view(B, C*4, H//2, W//2)

    
    def idwt(self, LL, LH, HL, HH):
        
        B, C, H, W = LL.shape
        LL = LL.reshape(B * C, 1, H, W)
        LH = LH.reshape(B * C, 1, H, W)
        HL = HL.reshape(B * C, 1, H, W)
        HH = HH.reshape(B * C, 1, H, W)
        
        out = torch.zeros(B * C, 1, H * 2, W * 2, device=LL.device)
        out[:, :, 0::2, 0::2] = (LL - LH - HL + HH) / np.sqrt(2)
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / np.sqrt(2)
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / np.sqrt(2)
        out[:, :, 1::2, 1::2] = (LL + LH + HL + HH) / np.sqrt(2)
        
        return out.view(B, C, H*2, W*2)
