import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def single_out(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.Sigmoid()
    )
def single_out1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

def single_out2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.Sigmoid()
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout=nn.Dropout(0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up31 = single_out1(768, 512)
        self.dconv_up32 = single_out1(512, 256)
        self.dconv_up21 = single_out1(384, 256)
        self.dconv_up22 = single_out1(256, 128)
        self.dconv_up11 = single_out1(192, 192)
        self.dconv_up12 = single_out1(192, 64)

        self.out = single_out(64, 3)
        self.out_conv1=single_out2(1,1)
        self.out_conv2 = single_out2(1, 1)
        self.out_conv3 = single_out2(1, 1)


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
        x = self.dropout(x)

        x = self.dconv_up31(x)
        x = self.dconv_up32(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dropout(x)

        x = self.dconv_up21(x)
        x = self.dconv_up22(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dropout(x)

        x = self.dconv_up11(x)
        x = self.dconv_up12(x)

        x = self.out(x)
        out1=x[:,0,:,:]
        out2 = x[:,1,:,:]
        out3 = x[:,2,:,:]

        out1=out1.unsqueeze(0)
        out2=out2.unsqueeze(0)
        out3=out3.unsqueeze(0)

        out1=self.out_conv1(out1)
        out2 = self.out_conv2(out2)
        out3 = self.out_conv3(out3)
        out=out1.add(out2)
        out=out.add(out3)


        return out1, out2, out3, out


output = UNet()

batch_size, C, H, W = 1, 1, 256, 256
x = torch.randn(batch_size, C, H, W)
out1, out2, out3, out = output(x)
