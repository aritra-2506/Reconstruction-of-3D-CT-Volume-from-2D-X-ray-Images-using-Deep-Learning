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
import pytorch_ssim
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import datetime, time

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

images=torch.from_numpy(images)
labels=torch.from_numpy(labels)

images_test1=images[2:4]
labels_test1=labels[2:4]

images_train=images[0:4]
images_val=images[0:2]
images_test=images[2:4]

labels_train=labels[0:4]
labels_val=labels[0:2]
labels_test=labels[2:4]

image_number=0
slice_number=120
image_app=images_test[image_number]
image_app = image_app.reshape((1, 1, 256, 256))
label_app=labels_test[image_number]
label_app = label_app.reshape((1, 150, 256, 256))

images=images.to(device)
labels=labels.to(device)

images_val=images_val.to(device)
labels_val=labels_val.to(device)

images_train=images_train.to(device)
images_val=images_val.to(device)
images_test=images_test.to(device)

labels_train=labels_train.to(device)
labels_val=labels_val.to(device)
labels_test=labels_test.to(device)

image_app=image_app.to(device)
label_app=label_app.to(device)

train_dataset = TensorDataset(images_train, labels_train)
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

test_dataset = TensorDataset(images_test, labels_test)
batch_size_test=1

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size_test
)


app_dataset = TensorDataset(image_app, label_app)
batch_size_app=1

app_loader = DataLoader(
    app_dataset,
    batch_size=batch_size_app
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
        x.to(device)

        return x


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


    for i, (images_train, labels_train) in enumerate(train_loader):

        optimizer.zero_grad()

        out = output(images_train)

        l = (out - labels_train)


        loss = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l**2))

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss + loss.item()

        epoch_loss=epoch_loss+loss.item()


        mse = criterion(out, labels_train)
        metric = 10 * math.log10(1 / mse.item())
        running_metric=running_metric+metric
        epoch_acc=epoch_acc+metric
        epoch_acc = epoch_acc + metric


        '''metric=pytorch_ssim.ssim(out, labels_train)

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
            out = output(images_val)

            l = (out - labels_val)

            val_loss = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))

            val_loss=val_loss.item()

            mse = criterion(out, labels_val)
            val_metric = 10 * math.log10(1 / mse.item())

            #val_metric = pytorch_ssim.ssim(out, labels_val)

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


running_test_loss = 0.0
running_test_metric = 0.0
n=1
test_loss_values=[]
test_metric_values=[]
batch_values=[]


with torch.set_grad_enabled(False):
    for i, (images_test, labels_test) in enumerate(
            test_loader):
        out = output(images_test)

        l = (out - labels_test)

        test_loss = 0.9 * torch.sum(torch.abs(l)) + 0.1 * torch.sqrt(torch.sum(l ** 2))

        running_test_loss = running_test_loss + test_loss.item()

        mse = criterion(out, labels_test)
        test_metric = 10 * math.log10(1 / mse.item())
        running_test_metric = running_test_metric + test_metric

        '''test_metric = pytorch_ssim.ssim(out, labels_test)
        running_test_metric = running_test_metric + test_metric.item()'''

        if (i % 1 == 0):    # print every 2 mini-batches
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



#app

def write_dicom(image2d, filename_little_endian, j):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.3.4"


    ds = FileDataset(filename_little_endian, {},
                     file_meta=meta, preamble=b"\0" * 128)

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"

    ds.Modality = "MR"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.ImagesInAcquisition = "1"

    ds.Rows = image2d.shape[0]
    ds.Columns = image2d.shape[1]
    ds.InstanceNumber = j

    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    print("Setting pixel data...")
    ds.PixelData = image2d.tobytes()
    ds.save_as(filename_little_endian)

    return

with torch.set_grad_enabled(False):
    for i, (image_app, label_app) in enumerate(
            app_loader):
        out, out1 = output(image_app)
        e = out.shape[1]
        for j in range(0, e):
            t = out[0][j].cpu().numpy()
            t = cv2.resize(t, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            filename_little_endian = ("/content/drive/My Drive/NewDataset/new_save_slices/%d.dcm" % (j,))
            ds = write_dicom(t, filename_little_endian, j)



dss1=pydicom.read_file("/content/drive/My Drive/NewDataset/new_save_slices/%d.dcm" % (slice_number,))
v=dss1.pixel_array

t=images_test1[image_number].reshape((256,256))
u=labels_test1[image_number][slice_number].reshape((256,256))

t=cv2.resize(np.float32(t), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
u=cv2.resize(np.float32(u), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

plt.figure()
plt.subplot(2, 2, 1)
plt.title('Original DRR')
plt.imshow(t, cmap=plt.cm.bone)
plt.subplot(2, 2, 2)
plt.title('Original Slice')
plt.imshow(u, cmap=plt.cm.bone)
plt.subplot(2, 2, 3)
plt.title('Decomposed Slice')
plt.imshow(v, cmap=plt.cm.bone)
