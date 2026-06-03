import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, dechannels, ratio=16):
        super().__init__()

        self.dechannels = dechannels
        self.mlp = nn.Sequential(
            nn.Linear(dechannels, dechannels // ratio),
            nn.ReLU(),
            nn.Linear(dechannels // ratio, dechannels)
        )
        self.mlp[-1].bias.data.zero_()

    def forward(self, x_de):

        B, C, _, _ = x_de.size()
        avg_pool = F.adaptive_avg_pool2d(x_de, 1).view(B, C)  
        max_pool = F.adaptive_max_pool2d(x_de, 1).view(B, C)
        
        avg_out = self.mlp(avg_pool) 
        max_out = self.mlp(max_pool)
        channel_weights = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        x_de = x_de * channel_weights + x_de   

        return x_de

class SpatialAttention(nn.Module):
   
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(
            2, 1, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x_en):
       
        avg_out = torch.mean(x_en, dim=1, keepdim=True)  
        max_out, _ = torch.max(x_en, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)  
        spatial_weights = self.conv(combined)                     
        spatial_weights = self.bn(spatial_weights)
        spatial_weights = torch.sigmoid(spatial_weights) 
        x_en = x_en * spatial_weights + x_en 
        
        return x_en
 
class AttentionConcat(nn.Module):
    def __init__(self, enchannels, dechannels):
        super().__init__()

        self.channelattention = ChannelAttention(dechannels)
        self.spatialattention = SpatialAttention()
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=enchannels + dechannels,
                out_channels=dechannels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(dechannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_en, x_de):

        x_en = self.spatialattention(x_en)  
        x_de = self.channelattention(x_de)  
        fused = torch.cat([x_en, x_de], dim=1)
        out = self.fuse(fused)

        return out 
    
