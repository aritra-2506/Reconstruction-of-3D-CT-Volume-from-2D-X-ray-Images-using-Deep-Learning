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

import pytorch_ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

'''PathDicom_W = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_W"
PathDicom_R = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_R"
PathDicom_S = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_S"
PathDicom_V = "C:/Users/Aritra Mazumdar/Downloads/ISIC/Dataset/Label_V"'''


PathDicom_W = "/content/drive/My Drive/hello/Dataset/Label_W"
PathDicom_R = "/content/drive/My Drive/hello/Dataset/Label_R"
PathDicom_S = "/content/drive/My Drive/hello/Dataset/Label_S"
PathDicom_V = "/content/drive/My Drive/hello/Dataset/Label_V"

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
images_no_aug = images.reshape((12, 1, 256, 256))
images_no_aug = images_no_aug.astype('float32') / 255

images_aug=images_no_aug[7:12]
images_no_aug1=images_no_aug[0:7]
d=len(images_aug)

images_list=[]
for i in range(0,d):
  h,w=images_aug[i][0].shape
  angle_range=(0, 180)
  angle = np.random.randint(*angle_range)
  images_augmentated = rotate(images_aug[0][0], angle)
  rate=0.5
  if np.random.rand() < rate:
      images_augmentated = images_augmentated[::-1, :]
  crop_size = (180, 180)
  top = (h - crop_size[0]) // 2
  left = (w - crop_size[1]) // 2
  bottom = top + crop_size[0]
  right = left + crop_size[1]
  images_augmentated = images_augmentated[top:bottom, left:right]
  images_augmentated = skimage.util.random_noise(images_augmentated, mode="gaussian")
  images_augmentated=cv2.resize(images_augmentated, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
  images_augmentated = images_augmentated.astype('float32')
  images_list.append(images_augmentated)

images_arr=np.asarray(images_list)
images_no_aug1 = images_no_aug1.reshape((7, 256, 256))
images_new=np.concatenate((images_no_aug1, images_arr), axis=0)
images_new = images_new.reshape((12, 1, 256, 256))
images=images_new


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

images_test1=images[10:12]

images=torch.from_numpy(images)
labels_D1=torch.from_numpy(labels_D1)
labels_D2=torch.from_numpy(labels_D2)
labels_D3=torch.from_numpy(labels_D3)


#Train-Test split
images_train=images[0:8]
images_val=images[8:10]
images_test=images[10:12]

labels_D1_train=labels_D1[0:8]
labels_D1_val=labels_D1[8:10]
labels_D1_test=labels_D1[10:12]

labels_D2_train=labels_D2[0:8]
labels_D2_val=labels_D2[8:10]
labels_D2_test=labels_D2[10:12]

labels_D3_train=labels_D3[0:8]
labels_D3_val=labels_D3[8:10]
labels_D3_test=labels_D3[10:12]

images=images.to(device)
images_train=images_train.to(device)
images_val=images_val.to(device)
images_test=images_test.to(device)

labels_D1_train=labels_D1_train.to(device)
labels_D1_val=labels_D1_val.to(device)
labels_D1_test=labels_D1_test.to(device)

labels_D2_train=labels_D2_train.to(device)
labels_D2_val=labels_D2_val.to(device)
labels_D2_test=labels_D2_test.to(device)

labels_D3_train=labels_D3_train.to(device)
labels_D3_val=labels_D3_val.to(device)
labels_D3_test=labels_D3_test.to(device)


train_dataset = TensorDataset(images_train, labels_D1_train, labels_D2_train, labels_D3_train)
batch_size=2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size
)

val_dataset = TensorDataset(images_val, labels_D1_val, labels_D2_val, labels_D3_val)
batch_size_val=images_val.shape[0]

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size_val
)

test_dataset = TensorDataset(images_test, labels_D1_test, labels_D2_test, labels_D3_test)
batch_size_test=2

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size_test-1
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

        self.out = single_out(64, 3)


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

        x=self.out(x)

        x=x.permute(1, 0, 2, 3)
        x1=x[0]
        x2=x[1]
        x3=x[2]
        x1=x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)

        out1 = x1.permute(1, 0, 2, 3)
        out2 = x2.permute(1, 0, 2, 3)
        out3 = x3.permute(1, 0, 2, 3)

        return out1, out2, out3


output = UNet()

output.cuda()

print(output)

optimizer = optim.Adam(output.parameters(), lr=0.001)

metric_values=[]
epoch_values=[]
loss_values=[]
val_loss_values=[]
val_metric_values=[]

criterion = nn.MSELoss()

for epoch in range(2):

    running_loss = 0.0
    running_metric = 0.0
    epoch_loss=0.0
    epoch_acc=0.0
    m=1


    for i, (images_train, labels_D1_train, labels_D2_train, labels_D3_train) in enumerate(train_loader):

        optimizer.zero_grad()

        out1, out2, out3 = output(images_train)

        l1 = (out1 - labels_D1_train)
        l2 = (out2 - labels_D2_train)
        l3 = (out3 - labels_D3_train)
        l = l1 + l2 + l3

        loss = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l**2))


        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss + loss.item()

        epoch_loss=epoch_loss+loss.item()
        #loss_total=running_loss/batch_size

        mse1 = criterion(out1, labels_D1_train)
        metric_1 = 10 * math.log10(1 / mse1.item())

        mse2 = criterion(out1, labels_D1_train)
        metric_2 = 10 * math.log10(1 / mse2.item())

        mse3 = criterion(out1, labels_D1_train)
        metric_3 = 10 * math.log10(1 / mse3.item())

        metric = (metric_1 + metric_2 + metric_3) / 3

        running_metric = running_metric + metric
        epoch_acc = epoch_acc + metric


        '''metric_1 = pytorch_ssim.ssim(out1, labels_D1_train)
        metric_2 = pytorch_ssim.ssim(out2, labels_D2_train)
        metric_3 = pytorch_ssim.ssim(out3, labels_D3_train)


        metric=(metric_1+metric_2+metric_3)/3


        running_metric = running_metric + metric.item()

        epoch_acc=epoch_acc+metric.item()'''

        if (i % 2 == 1):    # print every 2000 mini-batches
            a=round((running_metric / 2), 3)


            print('loss batch', m, 'epoch', epoch+1, ':', "%.3f" % round((running_loss/2), 3),'-','metric batch', m, 'epoch', epoch+1, ':', "%.3f" % round((running_metric/2), 3))

            running_loss = 0.0
            running_metric = 0.0
            m=m+1
        m=m
    output.eval()
    with torch.set_grad_enabled(False):
        for i, (images_val, labels_D1_val, labels_D2_val, labels_D3_val) in enumerate(
                val_loader):
            out1, out2, out3 = output(images_val)

            l1 = (out1 - labels_D1_val)
            l2 = (out2 - labels_D2_val)
            l3 = (out3 - labels_D3_val)
            l = l1 + l2 + l3

            val_loss = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))

            mse11 = criterion(out1, labels_D1_test)
            metric_11 = 10 * math.log10(1 / mse11.item())

            mse22 = criterion(out1, labels_D1_test)
            metric_22 = 10 * math.log10(1 / mse22.item())

            mse33 = criterion(out1, labels_D1_test)
            metric_33 = 10 * math.log10(1 / mse33.item())

            val_metric = (metric_11 + metric_22 + metric_33) / 3

            '''metric_11 = pytorch_ssim.ssim(out1, labels_D1_val)
            metric_22 = pytorch_ssim.ssim(out2, labels_D2_val)
            metric_33 = pytorch_ssim.ssim(out3, labels_D3_val)

            val_metric = (metric_11+metric_22+metric_33)/3'''



    m=m
    epoch_loss = epoch_loss / 4
    epoch_acc = epoch_acc / 4
    val_loss=val_loss
    val_metric=val_metric
    j=epoch+1

    print('loss epoch', j, ':', "%.3f" % round(epoch_loss, 3), '-', 'metric epoch',
              epoch + 1, ':', "%.3f" % round(epoch_acc, 3),'-', 'val loss epoch', j, ':', "%.3f" % val_loss, '-', 'val metric epoch',
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

running_test_loss = 0.0
running_test_metric = 0.0
n=1
test_loss_values=[]
test_metric_values=[]
batch_values=[]

outputs=[]
outputs1=[]
outputs2=[]
outputs3=[]

with torch.set_grad_enabled(False):
    for i, (images_test, labels_D1_test, labels_D2_test, labels_D3_test) in enumerate(
            test_loader):
        out1, out2, out3 = output(images_test)

        outputs1.append(out1)
        outputs2.append(out2)
        outputs3.append(out3)

        l1 = (out1 - labels_D1_test)
        l2 = (out2 - labels_D2_test)
        l3 = (out3 - labels_D3_test)
        l = l1 + l2 + l3

        test_loss = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))


        running_test_loss = running_test_loss + test_loss.item()

        mse111 = criterion(out1, labels_D1_test)
        metric_111 = 10 * math.log10(1 / mse111.item())

        mse222 = criterion(out1, labels_D1_test)
        metric_222 = 10 * math.log10(1 / mse222.item())

        mse333 = criterion(out1, labels_D1_test)
        metric_333 = 10 * math.log10(1 / mse333.item())

        test_metric = (metric_111 + metric_222 + metric_333) / 3

        running_test_metric = running_test_metric + test_metric
        


        '''metric_111 = pytorch_ssim.ssim(out1, labels_D1_train)
        metric_222 = pytorch_ssim.ssim(out2, labels_D2_train)
        metric_333 = pytorch_ssim.ssim(out3, labels_D3_train)

        test_metric = (metric_111 + metric_222 + metric_333) / 3

        running_test_metric = running_test_metric + test_metric.item()'''

        if (i % 1 == 0):    # print every 2000 mini-batches
            batch_values.append(n)
            test_loss_values.append(round((running_test_loss), 3))
            test_metric_values.append(round((running_test_metric), 3))


            print('test loss batch', n, ':', "%.3f" % round((running_test_loss), 3),'-','test metric batch', n, 'epoch', ':', "%.3f" % round((running_test_metric), 3))

            running_test_loss = 0.0
            running_test_metric = 0.0
            n=n+1


f=plt.figure(1)
plt.title('Model Test Loss')
plt.ylabel('Loss')
plt.xlabel('Number of batches')
plt.plot(batch_values, test_loss_values,'r')
f.show()

g=plt.figure(2)
plt.title('Model Test SSIM')
plt.ylabel('SSIM')
plt.xlabel('Number of batches')
plt.plot(batch_values, test_metric_values,'b')
g.show()


h=len(outputs1)

outputs_list1=[]
outputs_list2=[]
outputs_list3=[]

for i in range(0,h):

  outputs_b = outputs1[i].tolist()
  outputs_c = outputs2[i].tolist()
  outputs_d = outputs3[i].tolist()

  outputs_list1.append(outputs_b)
  outputs_list2.append(outputs_c)
  outputs_list3.append(outputs_d)


outputs_np1=np.asarray(outputs_list1)
outputs_np2=np.asarray(outputs_list2)
outputs_np3=np.asarray(outputs_list3)



for i in range(0, h):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Original DRR')
    plt.imshow(images_test1[i].reshape((256,256)))
    plt.subplot(2, 2, 2)
    plt.title('DRR Ribs')
    plt.imshow(outputs_np1[i].reshape((256,256)))
    plt.subplot(2, 2, 3)
    plt.title('DRR Vascular')
    plt.imshow(outputs_np2[i].reshape((256,256)))
    plt.subplot(2, 2, 4)
    plt.title('DRR Spine')
    plt.imshow(outputs_np3[i].reshape((256,256)))
    plt.show()

