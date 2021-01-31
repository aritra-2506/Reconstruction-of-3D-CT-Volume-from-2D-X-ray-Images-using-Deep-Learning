import torch.nn as nn
import torch

#network

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

def single_out1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

def single_out(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.Sigmoid()
    )

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(3, 300)
        self.dconv_down2 = double_conv(300, 512)
        self.dconv_down3 = double_conv(512, 1024)
        self.dconv_down4 = double_conv(1024, 2048)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up31 = single_out1(3072, 2048)
        self.dconv_up32 = single_out1(2048, 1024)
        self.dconv_up21 = single_out1(1536, 1024)
        self.dconv_up22 = single_out1(1024, 512)
        self.dconv_up11 = single_out1(812, 512)
        self.dconv_up12 = single_out1(512, 300)
        self.dconv = single_out(300, 256)
        self.dconv1 = single_out1(1, 1)
        self.dconv2 = single_out(1, 3)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.dropout(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up31(x)
        x = self.dconv_up32(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up21(x)
        x = self.dconv_up22(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up11(x)
        x = self.dconv_up12(x)

        x = self.dconv(x)

        out_1 = x

        i = out_1.shape[0]
        j = out_1.shape[1]

        for u in range(i):
            out_1_1 = out_1[u]
            for v in range(j):
                out_1_1_1 = out_1_1[v]
                out_1_1_1 = out_1_1_1.unsqueeze(0)
                out_1_1_1 = out_1_1_1.unsqueeze(0)
                out_1_1_1 = self.dconv1(out_1_1_1)
                if (v == 0):
                    out_2_1_1 = out_1_1_1
                else:
                    out_2_1_1 = torch.cat([out_2_1_1, out_1_1_1], dim=1)
            out_2_1 = torch.sum(out_2_1_1, dim=1)
            out_2_1 = out_2_1.unsqueeze(0)
            out_2_1 = self.dconv2(out_2_1)
            if (u == 0):
                out_2 = out_2_1
            else:
                out_2 = torch.cat([out_2, out_2_1], dim=0)

        return out_1, out_2

