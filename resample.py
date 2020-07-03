import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import math
from scipy.ndimage.interpolation import rotate
import skimage
from scipy.ndimage import zoom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import itertools
import os

PathDicom_W = "/content/drive/My Drive/NewDataset/resave"
#NewPathDicom_W="C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/backup"

#PathDicom_W = "C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/new_save"

ds2 = pydicom.read_file("/content/drive/My Drive/NewDataset/resave/2.dcm")
ds1 = pydicom.read_file("/content/drive/My Drive/NewDataset/resave/1.dcm")
sliceThickness = ds1.SliceLocation - ds2.SliceLocation
slice1Location = ds1.SliceLocation

dss=pydicom.read_file("/content/drive/My Drive/NewDataset/resave/1.dcm")


lstFilesDCM_W = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom_W):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM_W.append(os.path.join(dirName,filename))

images=[]

for filenameDCM in lstFilesDCM_W:
    # read the file
    ds = pydicom.read_file(filenameDCM)
    #print(length_of_pixel_array)
    x=cv2.resize(ds.pixel_array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # store the raw image data
    images.append(x)

images=np.asarray(images)

def write_dicom(image2d, i, slicesLocation, dss):

    ds=dss
    ds.InstanceNumber = i + 1

    ds.file_meta.MediaStorageSOPInstanceUID=pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()

    sliceLocation = float(slicesLocation)
    ds.sliceLocation = sliceLocation

    ds.PixelSpacing = r"0.822266\0.822266"
    print(ds.sliceLocation)
    a = float(-234.800003)

    b = float(-170.500000)

    print(a)
    ds.ImagePositionPatient = [a, b, sliceLocation]

    print("Setting pixel data...")
    ds.PixelData = image2d.tobytes()

    return ds


z=images.shape[0]
y=150/z


images_new = zoom(images, (y, 1, 1))

j=images_new.shape[0]



for i in range(0,j):
    a= images_new[i]
    slicesLocation = float(slice1Location - (sliceThickness * i))
    ds = write_dicom(a, i, slicesLocation, dss)
    ds.save_as(r"/content/drive/My Drive/NewDataset/aka/%d.dcm"%(i+1,))





