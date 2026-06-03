import torch
import torch.nn as nn
import torch.nn.functional as F
from Wavelet.wavelettransform import WaveletTransform
from Wavelet.subunet import *
from Wavelet.UNetattention import AttentionConcat
from Wavelet.manifold import GPUManifold
from thop import profile

class CONVF(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(CONVF, self).__init__()
        self.CONVF = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.CONVF(x)

class WaveletLHSeparable2(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.wavelet = WaveletTransform()
        self.hh_net = GPUManifold(in_channels)
        self.low_net = SubUNet2(in_channels)
        self.post_conv = CONVF(in_channels, out_channels)
        
    def forward(self, x):
      
        dwt_result = self.wavelet.dwt(x)
        B, C4, H, W = dwt_result.shape
        C = C4 // 4
        
        LL = dwt_result[:, :C, :, :]      
        LH = dwt_result[:, C:2*C, :, :]   
        HL = dwt_result[:, 2*C:3*C, :, :] 
        HH = dwt_result[:, 3*C:, :, :]    
 
        processed_HH, loss_hh = self.hh_net(HH)
        processed_L = self.low_net(torch.cat([LL, LH, HL], dim=1))
        LL_recon, LH_recon, HL_recon = torch.chunk(processed_L, 3, dim=1)
        reconstruct = self.wavelet.idwt(LL_recon, LH_recon, HL_recon, processed_HH)
        reconstruct = reconstruct + x
    
        return self.post_conv(reconstruct), loss_hh

class BottleneckWaveletLHSeparabledilation12(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.wavelet = WaveletTransform()
        self.hh_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.low_net = SubUNetdilation12(in_channels)
       
        self.post_conv = CONVF(in_channels, out_channels)

    def forward(self, x):
        
        dwt_result = self.wavelet.dwt(x)
        B, C4, H, W = dwt_result.shape
        C = C4 // 4

        LL = dwt_result[:, :C, :, :]     
        LH = dwt_result[:, C:2*C, :, :]   
        HL = dwt_result[:, 2*C:3*C, :, :] 
        HH = dwt_result[:, 3*C:, :, :]    

        processed_HH = self.hh_conv(HH)
        processed_L = self.low_net(torch.cat([LL, LH, HL], dim=1))
        LL_recon, LH_recon, HL_recon = torch.chunk(processed_L, 3, dim=1)
        reconstruct = self.wavelet.idwt(LL_recon, LH_recon, HL_recon, processed_HH)
        reconstruct = reconstruct + x

        return self.post_conv(reconstruct)
    
class DecodeWaveletLHSeparable2(nn.Module):
  
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.wavelet = WaveletTransform()
        self.hh_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.low_net = SubUNet2(in_channels)
        self.post_conv = CONVF(in_channels, out_channels)

    def forward(self, x):

        dwt_result = self.wavelet.dwt(x)
        B, C4, H, W = dwt_result.shape
        C = C4 // 4

        LL = dwt_result[:, :C, :, :]     
        LH = dwt_result[:, C:2*C, :, :]  
        HL = dwt_result[:, 2*C:3*C, :, :] 
        HH = dwt_result[:, 3*C:, :, :]   

        processed_HH = self.hh_conv(HH)
        processed_L = self.low_net(torch.cat([LL, LH, HL], dim=1))
        LL_recon, LH_recon, HL_recon = torch.chunk(processed_L, 3, dim=1)
        reconstruct = self.wavelet.idwt(LL_recon, LH_recon, HL_recon, processed_HH)
        reconstruct = reconstruct + x

        return self.post_conv(reconstruct)

def upsample(src,tar):

        src = F.upsample(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

        return src

class WaveletUNetLHSeparable(nn.Module):
 
    def __init__(self, in_channels=1, out_channels=1, mode='train', deepsuper=True):
        super().__init__()
        
        self.deepsuper = deepsuper  
        self.input = CONVF(in_channels, 32)
        self.enc1 = WaveletLHSeparable2(32, 64)
        self.pool2 = nn.MaxPool2d(2) 
        self.enc2 = WaveletLHSeparable2(64, 128)
        self.pool3 = nn.MaxPool2d(2) 
        self.enc3 = WaveletLHSeparable2(128, 256)
        self.pool4 = nn.MaxPool2d(2) 
        self.enc4 = WaveletLHSeparable2(256, 512)
        self.pool = nn.MaxPool2d(2) 
        self.bottleneck = BottleneckWaveletLHSeparabledilation12(512, 512)   
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attentionconcat4 = AttentionConcat(512, 512)
        self.dec4 = DecodeWaveletLHSeparable2(512, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attentionconcat3 = AttentionConcat(256, 256)
        self.dec3 = DecodeWaveletLHSeparable2(256, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) 
        self.attentionconcat2 = AttentionConcat(128, 128)
        self.dec2 = DecodeWaveletLHSeparable2(128, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attentionconcat1 = AttentionConcat(64, 64)
        self.dec1 = DecodeWaveletLHSeparable2(64, 32)

        self.out_convb = nn.Conv2d(512, out_channels, 1) 
        self.out_conv4 = nn.Conv2d(256, out_channels, 1) 
        self.out_conv3 = nn.Conv2d(128, out_channels, 1) 
        self.out_conv2 = nn.Conv2d(64, out_channels, 1)   
        self.out_conv1 = nn.Conv2d(32, out_channels, 1)   
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 5, padding=2)
        )
        
        self.final_conv = nn.Conv2d(6, out_channels, 1)
    
    def forward(self, x):

        x_in = self.input(x) 
        e1, eloss1 = self.enc1(x_in)   
       
        e21 = self.pool2(e1)
        e22, eloss2 = self.enc2(e21)   
       
        e31 = self.pool3(e22)
        e32, eloss3 = self.enc3(e31)  

        e41 = self.pool3(e32)
        e42, eloss4 = self.enc4(e41)  

        b1 = self.pool(e42)
        b2 = self.bottleneck(b1)

        d41 = self.up4(b2)
        d42 = self.dec4(self.attentionconcat4(e42, d41))
      
        d31 = self.up3(d42)
        d32 = self.dec3(self.attentionconcat3(e32, d31))
        
        d21 = self.up2(d32)
        d22 = self.dec2(self.attentionconcat2(e22, d21))
      
        d11 = self.up1(d22)
        d12 = self.dec1(self.attentionconcat1(e1, d11))  
    
        outb = self.out_convb(b2)
        outb = F.interpolate(outb, x.size()[2:], mode='bilinear', align_corners=False)
        out4 = self.out_conv4(d42)
        out4 = F.interpolate(out4, x.size()[2:], mode='bilinear', align_corners=False)
        out3 = self.out_conv3(d32)
        out3 = F.interpolate(out3, x.size()[2:], mode='bilinear', align_corners=False)
        out2 = self.out_conv2(d22)
        out2 = F.interpolate(out2, x.size()[2:], mode='bilinear', align_corners=False)
        out1 = self.out_conv1(d12)
        out = self.out_conv(d12)
        out = self.final_conv(torch.cat((out, out1, out2, out3, out4, outb), 1))
        
        if not self.training:

            return F.sigmoid(out), None

        loss_manifold = eloss1 + eloss2 + eloss3 + eloss4

        return F.sigmoid(out), loss_manifold


