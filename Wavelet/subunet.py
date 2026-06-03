import torch
import torch.nn as nn
import torch.nn.functional as F

class ENCONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ENCONV,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

class DECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DECONV,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

class ConvReduce(nn.Module):
    def __init__(self, C, bias=False):
        super(ConvReduce, self).__init__()
        self.conv = nn.Conv2d(in_channels=3*C,
                              out_channels=C,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=bias)

    def forward(self, x):
        return self.conv(x) 

class ConvExpand(nn.Module):
    def __init__(self, C, bias=False):
        super(ConvExpand, self).__init__()
        self.conv = nn.Conv2d(in_channels=C,
                              out_channels=3*C,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=bias)

    def forward(self, x):
        return self.conv(x) 

class CONVdilation12(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super(CONVdilation12, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1), #dilation=1
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),  #dilation=2
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SubUNetdilation12(nn.Module):
    def __init__(self, in_ch):
        super(SubUNetdilation12, self).__init__()
        self.reduce = ConvReduce(in_ch)
        self.en1 = ENCONV(in_ch, in_ch)
        self.down1 = nn.MaxPool2d(2)
        self.bn = CONVdilation12(in_ch, in_ch)
        self.de1 = DECONV(in_ch * 2, in_ch)
        self.expend = ConvExpand(in_ch)
    def forward(self, x):
        x_reduce = self.reduce(x)
        en1 = self.en1(x_reduce)          
        e1 = self.down1(en1)
        bn = self.bn(e1)
        d1 = F.interpolate(bn, size=en1.shape[2:], 
                            mode='bilinear', align_corners=False)
        cat = torch.cat([d1, en1], dim=1)
        de1 = self.de1(cat)
        out = self.expend(de1)
        out = out + x
        return out

class SubUNet2(nn.Module):
    def __init__(self, in_ch):
        super(SubUNet2, self).__init__()
        self.reduce = ConvReduce(in_ch)
        self.en1 = ENCONV(in_ch, in_ch)
        self.down1 = nn.MaxPool2d(2)
        self.bn = ENCONV(in_ch, in_ch)
        self.de1 = DECONV(in_ch * 2, in_ch)
        self.expend = ConvExpand(in_ch)
    def forward(self, x):
        x_reduce = self.reduce(x)
        en1 = self.en1(x_reduce)          
        e1  = self.down1(en1)      
        bn  = self.bn(e1)         
        d1  = F.interpolate(bn, size=en1.shape[2:], 
                            mode='bilinear', align_corners=False)  
        cat = torch.cat([d1, en1], dim=1)  
        de1 = self.de1(cat)   
        out = self.expend(de1)
        out = out + x                          
        return out


