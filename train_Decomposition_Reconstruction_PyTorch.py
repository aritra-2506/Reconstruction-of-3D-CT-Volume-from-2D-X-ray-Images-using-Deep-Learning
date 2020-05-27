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

images_test1=images[10:12]

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

labels_R_train=labels_R[0:8]
labels_R_val=labels_R[8:10]
labels_R_test=labels_R[10:12]



train_dataset = TensorDataset(images_train, labels_D1_train, labels_D2_train, labels_D3_train, labels_R_train)
batch_size=2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size
)

val_dataset = TensorDataset(images_val, labels_D1_val, labels_D2_val, labels_D3_val, labels_R_val)
batch_size_val=images_val.shape[0]

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size_val
)

test_dataset = TensorDataset(images_test, labels_D1_test, labels_D2_test, labels_D3_test, labels_R_test)
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

        x=self.out(x)

        x=x.permute(1, 0, 2, 3)
        x1=x[0]
        x2=x[1]
        x3=x[2]
        x1=x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        x3 = x3.unsqueeze(0)

        x1 = x1.permute(1, 0, 2, 3)
        x2 = x2.permute(1, 0, 2, 3)
        x3 = x3.permute(1, 0, 2, 3)


        out1=self.out_conv1(x1)
        out2 = self.out_conv2(x2)
        out3 = self.out_conv3(x3)
        out=out1.add(out2)
        out=out.add(out3)


        return out1, out2, out3, out


output = UNet()

print(output)

optimizer = optim.Adam(output.parameters(), lr=0.001)

metric_values=[]
epoch_values=[]
loss_values=[]
val_loss_values=[]
val_metric_values=[]

for epoch in range(2):

    running_loss = 0.0
    running_metric = 0.0
    epoch_loss=0.0
    epoch_acc=0.0
    m=1


    for i, (images_train, labels_D1_train, labels_D2_train, labels_D3_train, labels_R_train) in enumerate(train_loader):

        optimizer.zero_grad()

        out1, out2, out3, out = output(images_train)

        l1 = (out1 - labels_D1_train)
        l2 = (out2 - labels_D2_train)
        l3 = (out3 - labels_D3_train)
        l = l1 + l2 + l3

        loss1 = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l**2))

        loss2 = torch.sqrt(torch.sum(torch.abs(out - labels_R_train)))

        loss = (loss1 + 0.5 * loss2)

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss + loss.item()

        epoch_loss=epoch_loss+loss.item()
        #loss_total=running_loss/batch_size


        '''max_pixel = 1.0
        metric=(10.0 * torch.log((max_pixel ** 2) / (torch.mean((out - labels_R_train)**2)))) / 2.303
        running_metric=running_metric+metric.item()'''

        k1 = 0.01
        k2 = 0.03
        max_val = 255
        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2

        a = ((2 * (torch.mean(out)) * (torch.mean(labels_R_train)) + c1) * ((2 * (
            torch.mean(torch.mul(out,labels_R_train))) - (torch.mean(out)) * (torch.mean(labels_R_train))) + c2))
        b = (((torch.mean(out))**2 + (torch.mean(labels_R_train))**2 + c1) * (
                    (torch.var(out))**2 + (torch.var(labels_R_train))**2 + c2))
        metric = (a / b)
        running_metric = running_metric + metric.item()
        #metric_values.append(metric)
        epoch_acc=epoch_acc+metric.item()

        if (i % 2 == 1):    # print every 2000 mini-batches
            a=round((running_metric / 2), 3)


            print('loss batch', m, 'epoch', epoch+1, ':', "%.3f" % round((running_loss/2), 3),'-','metric batch', m, 'epoch', epoch+1, ':', "%.3f" % round((running_metric/2), 3))

            running_loss = 0.0
            running_metric = 0.0
            m=m+1
        m=m

    with torch.set_grad_enabled(False):
        for i, (images_val, labels_D1_val, labels_D2_val, labels_D3_val, labels_R_val) in enumerate(
                val_loader):
            out1, out2, out3, out = output(images_val)

            l1 = (out1 - labels_D1_val)
            l2 = (out2 - labels_D2_val)
            l3 = (out3 - labels_D3_val)
            l = l1 + l2 + l3

            val_loss1 = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))

            val_loss2 = torch.sqrt(torch.sum(torch.abs(out - labels_R_train)))

            val_loss = (val_loss1 + 0.5 * val_loss2)

            '''max_pixel = 1.0
            metric=(10.0 * torch.log((max_pixel ** 2) / (torch.mean((out - labels_R_val)**2)))) / 2.303'''

            k1 = 0.01
            k2 = 0.03
            max_val = 255
            c1 = (k1 * max_val) ** 2
            c2 = (k2 * max_val) ** 2

            a = ((2 * (torch.mean(out)) * (torch.mean(labels_R_val)) + c1) * ((2 * (
                torch.mean(torch.mul(out, labels_R_val))) - (torch.mean(out)) * (torch.mean(labels_R_val))) + c2))
            b = (((torch.mean(out)) ** 2 + (torch.mean(labels_R_val)) ** 2 + c1) * (
                    (torch.var(out)) ** 2 + (torch.var(labels_R_val)) ** 2 + c2))
            val_metric = (a / b)



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
    for i, (images_test, labels_D1_test, labels_D2_test, labels_D3_test, labels_R_test) in enumerate(
            test_loader):
        out1, out2, out3, out = output(images_test)
        outputs.append(out)
        outputs1.append(out1)
        outputs2.append(out2)
        outputs3.append(out3)

        l1 = (out1 - labels_D1_test)
        l2 = (out2 - labels_D2_test)
        l3 = (out3 - labels_D3_test)
        l = l1 + l2 + l3

        test_loss1 = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))

        test_loss2 = torch.sqrt(torch.sum(torch.abs(out - labels_R_test)))

        test_loss = (test_loss1 + 0.5 * test_loss2)
        running_test_loss = running_test_loss + test_loss.item()

        '''max_pixel = 1.0
        metric=(10.0 * torch.log((max_pixel ** 2) / (torch.mean((out - labels_R_val)**2)))) / 2.303
        running_test_metric=running_test_metric+metric.item()'''

        k1 = 0.01
        k2 = 0.03
        max_val = 255
        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2

        a = ((2 * (torch.mean(out)) * (torch.mean(labels_R_test)) + c1) * ((2 * (
            torch.mean(torch.mul(out, labels_R_test))) - (torch.mean(out)) * (torch.mean(labels_R_test))) + c2))
        b = (((torch.mean(out)) ** 2 + (torch.mean(labels_R_test)) ** 2 + c1) * (
                (torch.var(out)) ** 2 + (torch.var(labels_R_test)) ** 2 + c2))
        test_metric = (a / b)
        running_test_metric = running_test_metric + test_metric.item()

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


h=len(outputs)
outputs_list=[]
outputs_list1=[]
outputs_list2=[]
outputs_list3=[]

for i in range(0,h):
  outputs_a=outputs[i].tolist()
  outputs_b = outputs1[i].tolist()
  outputs_c = outputs2[i].tolist()
  outputs_d = outputs3[i].tolist()
  outputs_list.append(outputs_a)
  outputs_list1.append(outputs_b)
  outputs_list2.append(outputs_c)
  outputs_list3.append(outputs_d)

outputs_np=np.asarray(outputs_list)
outputs_np1=np.asarray(outputs_list1)
outputs_np2=np.asarray(outputs_list2)
outputs_np3=np.asarray(outputs_list3)



for i in range(0, h):
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.title('Original DRR')
    plt.imshow(images_test1[i].reshape((256,256)))
    plt.subplot(2, 3, 2)
    plt.title('DRR Ribs')
    plt.imshow(outputs_np1[i].reshape((256,256)))
    plt.subplot(2, 3, 3)
    plt.title('DRR Vascular')
    plt.imshow(outputs_np2[i].reshape((256,256)))
    plt.subplot(2, 3, 4)
    plt.title('DRR Spine')
    plt.imshow(outputs_np3[i].reshape((256,256)))
    plt.subplot(2, 3, 5)
    plt.title('DRR Reconstructed')
    plt.imshow(outputs_np[i].reshape((256,256)))
    plt.show()

