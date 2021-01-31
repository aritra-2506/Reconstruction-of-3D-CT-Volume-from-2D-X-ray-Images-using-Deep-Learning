import glob
import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt


def main(folderPath):

    my_glob = glob.glob1(folderPath, "*")
    if len(my_glob) == 0:
        return False

    # get all readable dicom files in  array
    tem = []
    for file in list(my_glob):
        data_item = pydicom.dcmread(os.path.join(folderPath, file))
        tem.append(data_item)

    if len(tem) <= 0:
        return False

    tem.sort(key=lambda x: x.InstanceNumber)

    # make 3d np array from all slices
    unset = True
    for i in range(len(tem)):
        arr = tem[i].pixel_array.astype(np.float32)
        if unset:
            imShape = (arr.shape[0], arr.shape[1], len(tem))
            scaledIm = np.zeros(imShape)
            pix_spacing = tem[i].PixelSpacing
            dist = 0
            for j in range(2):
                cs = [float(q) for q in tem[j].ImageOrientationPatient]
                ipp = [float(q) for q in tem[j].ImagePositionPatient]
                parity = pow(-1, j)
                dist += parity * (cs[1] * cs[5] - cs[2] * cs[4]) * ipp[0]
                dist += parity * (cs[2] * cs[3] - cs[0] * cs[5]) * ipp[1]
                dist += parity * (cs[0] * cs[4] - cs[1] * cs[3]) * ipp[2]
            z_spacing = abs(dist)
            slope = tem[i].RescaleSlope
            intercept = tem[i].RescaleIntercept
            unset = False
        scaledIm[:, :, i] = arr

    # convert to hounsfield units
    scaledIm = slope * scaledIm + intercept
    pix_spacing.append(z_spacing)

    wl = 300  # window parameters for Angio
    ww = 600

    windowed = np.zeros(imShape, dtype=np.uint8)
    # allImages[scaledIm <= (wl-0.5-(ww-1)/2.0)] = 0
    k = np.logical_and(scaledIm > (wl - 0.5 - (ww - 1) / 2.0), scaledIm <= (wl - 0.5 + (ww - 1) / 2.0))
    windowed[k] = ((scaledIm[k] - (wl - 0.5)) / (ww - 1) + 0.5) * 255
    windowed[scaledIm > (wl - 0.5 + (ww - 1) / 2.0)] = 255


    windowed = np.transpose(windowed)


    # Creating 2D DRR Label Data for Original and Sliced Volumes
    a = windowed.sum(2)

    return a

def my_transpose(folder):
    a = main(folder)
    return a