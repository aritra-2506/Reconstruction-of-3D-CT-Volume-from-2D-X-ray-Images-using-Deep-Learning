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
from scipy.ndimage.interpolation import rotate
import skimage
from scipy.ndimage import zoom

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

'''PathDicom_images = "C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/NewDataset/images"
PathDicom_labels = "C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/NewDataset/labels"'''

PathDicom_images = "/content/drive/My Drive/NewDataset/images"
PathDicom_labels = "/content/drive/My Drive/NewDataset/labels"

images=[]
for dirName, subdirList, fileList in os.walk(PathDicom_images):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM

            ds = pydicom.read_file(os.path.join(dirName, filename))
            x = cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            # store the raw image data
            images.append(x)

images=np.asarray(images)
images = cv2.normalize(images, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
images.astype(np.uint8)
images = images.reshape((4, 1, 256, 256))
images = images.astype('float32') / 255


labels_whole = []
for dirName, subdirList, fileList in os.walk(PathDicom_labels):
        for filename in fileList:

            if ".dcm" in filename.lower():  # check whether the file's DICOM

                ds = pydicom.read_file(os.path.join(dirName, filename))
                x = cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                # store the raw image data
                labels_whole.append(x)




labels_whole=np.asarray(labels_whole)

j=labels_whole.shape[0]

for i in range(0,j):
    if(i==0):
        x=labels_whole[0:150]
        x=x.reshape((1, 150, 256, 256))
    elif(i%150==0):
        y = labels_whole[i:i + 150]
        y=y.reshape((1, 150, 256, 256))
        x=np.concatenate((x,y))
labels=x

labels = cv2.normalize(labels, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels.astype(np.uint8)
labels = labels.astype('float32') / 255

images_val=images[0:2]
labels_val=labels[0:2]

images=torch.from_numpy(images)
labels=torch.from_numpy(labels)

images_val=torch.from_numpy(images_val)
labels_val=torch.from_numpy(labels_val)

images=images.to(device)
labels=labels.to(device)

images_val=images_val.to(device)
labels_val=labels_val.to(device)

train_dataset = TensorDataset(images, labels)
batch_size=2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size
)

val_dataset = TensorDataset(images_val, labels_val)
batch_size_val=images_val.shape[0]

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size_val
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
        self.dropout = nn.Dropout(0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up31 = single_out1(768, 512)
        self.dconv_up32 = single_out1(512, 256)
        self.dconv_up21 = single_out1(384, 256)
        self.dconv_up22 = single_out1(256, 128)
        self.dconv_up11 = single_out1(192, 192)
        self.dconv_up12 = single_out1(192, 64)

        self.out = single_out(64, 150)
        self.out_conv1 = single_out2(1, 1)

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
        m = x.shape[1]
        convo = []

        for i in range(0, m):
            d = x.permute(1, 0, 2, 3)
            d = d[i]
            d = d.unsqueeze(0)
            d = d.permute(1, 0, 2, 3)
            d = self.out_conv1(d)
            d = d.permute(1, 0, 2, 3)
            convo.append(d)

        k = len(convo)
        for i in range(0, k):
            if (i == 0):
                g = convo[0]
                # g=g.unsqueeze(0)
            else:
                f = convo[i]
                # f = f.unsqueeze(0)
                g = torch.cat([g, f], dim=0)

        convs = g

        convs = convs.permute(1, 0, 2, 3)
        convs = torch.sum(convs, dim=3)
        convs = convs.unsqueeze(0)
        convs = convs.permute(1, 0, 2, 3)
        convs = convs.permute(0, 1, 3, 2)
        convs = torch.nn.functional.interpolate(convs, size=256)
        convs = convs.permute(0, 1, 3, 2)

        x.to(device)
        convs.to(device)

        return x, convs


output = UNet()

output.cuda()

# print(output)

optimizer = optim.Adam(output.parameters(), lr=0.000001)

metric_values = []
epoch_values = []
loss_values = []
val_loss_values = []
val_metric_values = []

criterion = nn.MSELoss()

for epoch in range(2):

    running_loss = 0.0
    running_metric = 0.0
    epoch_loss=0.0
    epoch_acc=0.0
    m=1


    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        out, out1 = output(images)

        l = (out - labels)


        loss1 = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l**2))
        loss2 = torch.sqrt(torch.sum(torch.abs(out1 - images)))
        loss = (loss1 + 0.5 * loss2)

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss + loss.item()

        epoch_loss=epoch_loss+loss.item()


        mse = criterion(out1, images)
        metric = 10 * math.log10(1 / mse.item())
        running_metric=running_metric+metric
        epoch_acc=epoch_acc+metric
        epoch_acc = epoch_acc + metric


        '''metric=pytorch_ssim.ssim(out, labels_R_train)

        running_metric = running_metric + metric.item()
        epoch_acc=epoch_acc+metric.item()'''

        if (i % 2 == 1):    # print every 2 mini-batches
            a=round((running_metric / 2), 3)


            print('loss batch', m, 'epoch', epoch+1, ':', "%.3f" % round((running_loss/2), 3),'-','metric batch', m, 'epoch', epoch+1, ':', "%.3f" % round((running_metric/2), 3))

            running_loss = 0.0
            running_metric = 0.0
            m=m+1
        m=m
    output.eval()

    with torch.set_grad_enabled(False):
        for i, (images_val, labels_val) in enumerate(
                val_loader):
            out, out1 = output(images_val)

            l = (out - labels_val)

            val_loss1 = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))

            val_loss2 = torch.sqrt(torch.sum(torch.abs(out1 - images_val)))

            val_loss = (val_loss1 + 0.5 * val_loss2)

            mse = criterion(out1, images_val)
            val_metric = 10 * math.log10(1 / mse.item())

            # val_metric = pytorch_ssim.ssim(out, labels_R_val)

    m = m
    epoch_loss = epoch_loss / 4
    epoch_acc = epoch_acc / 4
    val_loss = val_loss
    val_metric = val_metric
    j = epoch + 1

    print('loss epoch', j, ':', "%.3f" % round(epoch_loss, 3), '-', 'metric epoch',
          epoch + 1, ':', "%.3f" % round(epoch_acc, 3), '-', 'val loss epoch', j, ':', "%.3f" % val_loss, '-',
          'val metric epoch',
          epoch + 1, ':', "%.3f" % val_metric)

    metric_values.append(round(epoch_acc, 3))
    loss_values.append(round(epoch_loss, 3))
    val_loss_values.append(val_loss)
    val_metric_values.append(val_metric)
    epoch_values.append(j)

print('Finished Training')


fig1,ax1=plt.subplots()
ax1.set_title('Model Loss')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Number of epochs')
ax1.plot(epoch_values, loss_values)
ax1.plot(epoch_values, val_loss_values)
ax1.legend(['Train', 'Val'])


fig2,ax2=plt.subplots()
ax2.set_title('Model SSIM')
ax2.set_ylabel('SSIM')
ax2.set_xlabel('Number of epochs')
ax2.plot(epoch_values, metric_values)
ax2.plot(epoch_values, val_metric_values)
ax2.legend(['Train', 'Val'])
plt.show()


#Save checkpoint
checkpoint = {'output': UNet(),
          'state_dict': output.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'C:/Users/Aritra Mazumdar/Downloads/ISIC/checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    output = checkpoint['output']
    output.load_state_dict(checkpoint['state_dict'])
    for parameter in output.parameters():
        parameter.requires_grad = False

    output.eval()
    return output

output = load_checkpoint('C:/Users/Aritra Mazumdar/Downloads/ISIC/checkpoint.pth')

#Save model
torch.save(output, 'C:/Users/Aritra Mazumdar/Downloads/ISIC/output.pth')

output = torch.load('C:/Users/Aritra Mazumdar/Downloads/ISIC/output.pth')
output.eval()

