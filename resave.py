import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import cv2

PathDicom = "C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/3000534-58228"


#lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM


            ds = pydicom.read_file(os.path.join(dirName, filename))
            j=ds.InstanceNumber
            ds.save_as(r"C:/Users/Aritra Mazumdar/Downloads/ISIC/data_folder/resave/%d.dcm" % (j,))



