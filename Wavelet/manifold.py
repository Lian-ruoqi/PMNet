import torch
import torch.nn as nn
import torch.nn.functional as F
from Wavelet.simplifycudaumap import ParametricUMAPModule

class manifoldConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.manifoldConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):

        return self.manifoldConv(x)

class GPUManifold(nn.Module):
    def __init__(self, in_channels, neighbor_size=3):
        super().__init__()

        self.down = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.neighbor_size = neighbor_size
        self.pad_size = neighbor_size // 2
        self.padder   = nn.ReflectionPad2d(self.pad_size)
        self.manifolddecoding = torch.nn.Sequential(
            manifoldConv(in_channels)
        )
        in_dim    = neighbor_size**2 * in_channels 
        out_dim   = in_channels
        self.umap = ParametricUMAPModule(
            in_dim=in_dim,
            out_dim=out_dim,
        )

    def _extract_patches(self, x):
        B, C, H, W = x.shape
        padded  = self.padder(x)
        patches = padded.unfold(2, self.neighbor_size, 1).unfold(3, self.neighbor_size, 1) 
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() 

        return patches.view(B, H * W, -1) 

    def forward(self, x):

        x = self.down(x)
        B, C, H, W = x.shape
        patches = self._extract_patches(x) 
        patches_flat = patches.view(-1, patches.size(-1))  
        Z, umap_loss = self.umap(patches_flat) 
        Z = Z.view(B, H, W, -1).permute(0, 3, 1, 2) 
        out    = self.manifolddecoding(Z)
        output = self.up(out)

        return output, umap_loss 