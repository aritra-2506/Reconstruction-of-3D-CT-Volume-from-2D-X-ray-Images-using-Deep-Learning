import time
import glob
import pydicom
import numpy as np
from vtkplotter import Volume
import os
import matplotlib.pyplot as plt
import cv2


def main(folderPath):
    st = time.time()
    my_glob = glob.glob1(folderPath, "*")
    numFiles = 0
    rejected = 0

    # return if empty directory
    if len(my_glob) == 0:
        return False

    # get all readable dicom files in  array
    tem = []
    for file in list(my_glob):
        try:
            data_item = pydicom.dcmread(os.path.join(folderPath, file))
            if hasattr(data_item, 'SliceLocation'):
                tem.append(data_item)
                numFiles += 1
            else:
                rejected += 1
                print(file)
        except Exception as e:
            pass
    print("read done %s | %d files | %d rejected" % (time.time() - st, numFiles, rejected))
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

    # Data Slicing for Ribs, Vascular and Spine
    windowed_R = windowed[0:180]
    windowed_V = windowed[180:360]
    windowed_S = windowed[360:512]

    windowed_R = np.transpose(windowed_R)
    windowed_V = np.transpose(windowed_V)
    windowed_S = np.transpose(windowed_S)
    windowed = np.transpose(windowed)

    # Zero Padding Sliced Data for Original Volume
    m = np.zeros((149, 512))
    j = m
    l = m
    p=m
    q=m
    for x in range(0, 331):
        j = np.dstack((j, m))
    for x in range(0, 359):
        q = np.dstack((q, m))
    for y in range(0, 179):
        l = np.dstack((l, m))
    for y in range(0, 151):
        p = np.dstack((p, m))
    windowed_R = np.dstack((windowed_R, j))
    windowed_S = np.dstack((q, windowed_S))
    windowed_V = np.dstack((l, windowed_V))
    windowed_V = np.dstack((windowed_V, p))

    # 3D DICOM Data
    a = np.transpose(windowed)

    # Creating 2D DRR Label Data for Original and Sliced Volumes
    b = windowed.sum(2)
    c = windowed_R.sum(2)
    d = windowed_V.sum(2)
    e = windowed_S.sum(2)

    return a, b, c, d, e, pix_spacing

if __name__ == '__main__':
    folder = "C:/Users/Aritra Mazumdar/Downloads/ISIC/3000568-87015/"
    p, q, r, s, t, u = main(folder)
    labels_D0 = []
    labels_D1 = []
    labels_D2 = []
    labels_D3 = []

    # 3D DICOM Data Visualization
    vol = Volume(p, spacing=u)
    vol.permuteAxes(2, 1, 0).mirror("y")
    vol.show(bg="black")

    # 2D DRR Plots
    fig1 = plt.figure()
    # Complete DRR
    a1 = fig1.add_subplot(2, 2, 1)
    a1.set_title('DRR Whole')
    a1.imshow(q, cmap=plt.cm.bone, aspect=3)
    # DRR Ribs
    a2 = fig1.add_subplot(2, 2, 2)
    a2.set_title('DRR Ribs')
    a2.imshow(r, cmap=plt.cm.bone, aspect=3)
    # DRR Vascular
    a3 = fig1.add_subplot(2, 2, 3)
    a3.set_title('DRR Vascular')
    a3.imshow(s, cmap=plt.cm.bone, aspect=3)
    # DRR Spine
    a4 = fig1.add_subplot(2, 2, 4)
    a4.set_title('DRR Spine')
    a4.imshow(t, cmap=plt.cm.bone, aspect=3)

    #Generating Labels
    labels_D0.append(q)
    labels_D0 = np.asarray(labels_D0)

    labels_D1.append(r)
    labels_D1 = np.asarray(labels_D1)

    labels_D2.append(s)
    labels_D2 = np.asarray(labels_D2)

    labels_D3.append(t)
    labels_D3 = np.asarray(labels_D3)

    #Plotting DRR Labels
    fig2 = plt.figure()
    # DRR Ribs
    a11 = fig2.add_subplot(2, 2, 1)
    a11.set_title('DRR Whole')
    a11.imshow(labels_D0[0], cmap=plt.cm.bone, aspect=3)
    #DRR Ribs
    a22 = fig2.add_subplot(2, 2, 2)
    a22.set_title('DRR Ribs')
    a22.imshow(labels_D1[0], cmap=plt.cm.bone, aspect=3)
    # DRR Vascular
    a33 = fig2.add_subplot(2, 2, 3)
    a33.set_title('DRR Vascular')
    a33.imshow(labels_D2[0], cmap=plt.cm.bone, aspect=3)
    # DRR Spine
    a44 = fig2.add_subplot(2, 2, 4)
    a44.set_title('DRR Spine')
    a44.imshow(labels_D3[0], cmap=plt.cm.bone, aspect=3)

    plt.show()
