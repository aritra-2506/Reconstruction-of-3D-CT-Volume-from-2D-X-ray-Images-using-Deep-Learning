import time
import glob
import pydicom
import numpy as np
import vtkplotter
from vtkplotter import Volume
import sys, os
import matplotlib.pyplot as plt


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

    # Rotations
    windowed = np.rot90(windowed)
    windowed = np.rot90(windowed)
    windowed = np.rot90(windowed)

    # Data Slicing for Ribs, Vascular and Spine
    windowed = np.transpose(windowed)
    windowed_R = windowed[0:50]
    windowed_V = windowed[50:100]
    windowed_S = windowed[100:150]
    windowed_R = np.transpose(windowed_R)
    windowed_V = np.transpose(windowed_V)
    windowed_S = np.transpose(windowed_S)
    windowed = np.transpose(windowed)

    # Zero Padding Sliced Data for Original Volume
    m = np.zeros((512, 512))
    j = m
    l = m
    for x in range(0, 98):
        j = np.dstack((j, m))
    for y in range(0, 48):
        l = np.dstack((l, m))
    windowed_R = np.dstack((windowed_R, j))
    k = np.dstack((j, m))
    windowed_S = np.dstack((k, windowed_S))
    o = l
    o = np.dstack((o, m))
    windowed_V = np.dstack((o, windowed_V))
    windowed_V = np.dstack((windowed_V, l))

    # 3D DICOM Data
    a = windowed

    # Creating 2D DRR Label Data for Original and Sliced Volumes
    b = windowed.sum(2)
    c = windowed_R.sum(2)
    d = windowed_V.sum(2)
    e = windowed_S.sum(2)
    return a, b, c, d, e, pix_spacing


if __name__ == '__main__':
    folder = "C:/Users/Aritra Mazumdar/Downloads/ISIC/3000568-87015/"
    p, q, r, s, t, u = main(folder)

    # 3D DICOM Data Visualization
    vol = Volume(p, spacing=u)
    vol.permuteAxes(2, 1, 0).mirror("y")
    vol.show(bg="black")

    # 2D DRR Plots
    fig = plt.figure()
    # Complete DRR
    a1 = plt.subplot(2, 2, 1)
    a1.title.set_text('DRR Whole')
    a1.imshow(q, cmap=plt.cm.bone, aspect=1)
    # DRR Ribs
    a2 = plt.subplot(2, 2, 2)
    a2.title.set_text('DRR Ribs')
    a2.imshow(r, cmap=plt.cm.bone, aspect=1)
    # DRR Vascular
    a3 = plt.subplot(2, 2, 3)
    a3.title.set_text('DRR Vascular')
    a3.imshow(s, cmap=plt.cm.bone, aspect=1)
    # DRR Spine
    a4 = plt.subplot(2, 2, 4)
    a4.title.set_text('DRR Spine')
    a4.imshow(t, cmap=plt.cm.bone, aspect=1)
    plt.show()
