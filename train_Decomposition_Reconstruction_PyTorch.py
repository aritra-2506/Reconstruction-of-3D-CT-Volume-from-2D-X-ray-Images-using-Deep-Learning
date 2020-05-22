import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import math
import torch
import torchvision
from torch import optim
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


PathDicom_W = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_W"
PathDicom_R = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_R"
PathDicom_S = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_S"
PathDicom_V = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_V"

lstFilesDCM_W = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom_W):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_W.append(os.path.join(dirName,filename))

lstFilesDCM_R = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom_R):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_R.append(os.path.join(dirName,filename))

lstFilesDCM_V = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom_V):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_V.append(os.path.join(dirName, filename))

lstFilesDCM_S = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom_S):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_S.append(os.path.join(dirName,filename))

images=[]
labels_D1=[]
labels_D2=[]
labels_D3=[]

for filenameDCM in lstFilesDCM_W:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    x=cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # store the raw image data
    images.append(x)

for filenameDCM in lstFilesDCM_R:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    x=cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # store the raw image data
    labels_D1.append(x)

for filenameDCM in lstFilesDCM_V:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    x=cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # store the raw image data
    labels_D2.append(x)

for filenameDCM in lstFilesDCM_S:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    x=cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # store the raw image data
    labels_D3.append(x)

images=np.asarray(images)
images = cv2.normalize(images, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
images.astype(np.uint8)
images = images.reshape((12, 1, 256, 256))
images = images.astype('float32') / 255

labels_R=images

labels_D1=np.asarray(labels_D1)
labels_D1 = cv2.normalize(labels_D1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D1.astype(np.uint8)
labels_D1 = labels_D1.reshape((12, 1, 256, 256))
labels_D1 = labels_D1.astype('float32') / 255

labels_D2=np.asarray(labels_D2)
labels_D2 = cv2.normalize(labels_D2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D2.astype(np.uint8)
labels_D2 = labels_D2.reshape((12, 1, 256, 256))
labels_D2 = labels_D2.astype('float32') / 255

labels_D3=np.asarray(labels_D3)
labels_D3 = cv2.normalize(labels_D3, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D3.astype(np.uint8)
labels_D3 = labels_D3.reshape((12, 1, 256, 256))
labels_D3 = labels_D3.astype('float32') / 255

images=torch.from_numpy(images)
labels_D1=torch.from_numpy(labels_D1)
labels_D2=torch.from_numpy(labels_D2)
labels_D3=torch.from_numpy(labels_D3)
labels_R=torch.from_numpy(labels_R)

#Train-Test split
images_train=images[0:8]
images_test=images[8:12]
labels_D1_train=labels_D1[0:8]
labels_D1_test=labels_D1[8:12]
labels_D2_train=labels_D2[0:8]
labels_D2_test=labels_D1[8:12]
labels_D3_train=labels_D3[0:8]
labels_D3_test=labels_D3[8:12]
labels_R_train=labels_R[0:8]
labels_R_test=labels_D1[8:12]



train_dataset = TensorDataset(images_train, labels_D1_train, labels_D2_train, labels_D3_train, labels_R_train)
batch_size=4

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size
)

test_dataset = TensorDataset(images_test, labels_D1_test, labels_D2_test, labels_D3_test, labels_R_test)
batch_size=4

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size
)

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

        self.out = single_out(64, 1)
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

        x1 = self.out(x)
        x2 = self.out(x)
        x3 = self.out(x)


        out1=self.out_conv1(x1)
        out2 = self.out_conv2(x2)
        out3 = self.out_conv3(x3)
        out=out1.add(out2)
        out=out.add(out3)


        return out1, out2, out3, out


output = UNet()


optimizer = optim.Adam(output.parameters(), lr=0.001)

for epoch in range(2):
    running_loss1 = 0.0
    running_loss2 = 0.0
    #running_loss3 = 0.0
    for i, (images_train, labels_D1_train, labels_D2_train, labels_D3_train, labels_R_train) in enumerate(train_loader):

        optimizer.zero_grad()

        out1, out2, out3, out = output(images_train)

        l1 = (out1 - labels_D1_train)
        l2 = (out2 - labels_D2_train)
        l3 = (out3 - labels_D3_train)
        l = l1 + l2 + l3

        loss1 = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l**2))
        loss1.backward(retain_graph=True)

        running_loss1 = running_loss1 + loss1.item()

        loss2 = torch.sqrt(torch.sum(torch.abs(out - labels_R_train)))
        loss2.backward(retain_graph=True)
        optimizer.step()
        running_loss2 = running_loss2 + loss2.item()
        loss_total = (running_loss1 + 0.5 * running_loss2)/batch_size

        running_metric=0.0
        max_pixel = 1.0
        metric=(10.0 * torch.log((max_pixel ** 2) / (torch.mean((out - labels_R_train)**2)))) / 2.303
        running_metric=running_metric+metric.item()
        metric_total=running_metric/batch_size

        print('loss batch', i+1, 'epoch', epoch+1, ':', "%.3f" % round(loss_total, 3),'-','metric batch', i+1, 'epoch', epoch+1, ':', "%.3f" % round(metric_total, 3))

print('Finished Training')
