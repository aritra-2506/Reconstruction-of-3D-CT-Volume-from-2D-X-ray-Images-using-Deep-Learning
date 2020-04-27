import cv2
import glob
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input, Lambda, Reshape
import keras.backend as k
from keras.models import Model
import os
import matplotlib.pyplot as plt
import pydicom
import keras.layers as h

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
images = images.reshape((4, 256, 256, 1))
images = images.astype('float32') / 255

labels_R=images

labels_D1=np.asarray(labels_D1)
labels_D1 = cv2.normalize(labels_D1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D1.astype(np.uint8)
labels_D1 = labels_D1.reshape((4, 256, 256, 1))
labels_D1 = labels_D1.astype('float32') / 255

labels_D2=np.asarray(labels_D2)
labels_D2 = cv2.normalize(labels_D2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D2.astype(np.uint8)
labels_D2 = labels_D2.reshape((4, 256, 256, 1))
labels_D2 = labels_D2.astype('float32') / 255

labels_D3=np.asarray(labels_D3)
labels_D3 = cv2.normalize(labels_D3, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D3.astype(np.uint8)
labels_D3 = labels_D3.reshape((4, 256, 256, 1))
labels_D3 = labels_D3.astype('float32') / 255


''''#Train-Test split
images_train=images[0:3]
images_test=images[3:5]
labels_D1_train=labels_D1[0:3]
labels_D1_test=labels_D1[3:5]
labels_D2_train=labels_D2[0:3]
labels_D2_test=labels_D1[3:5]
labels_D3_train=labels_D3[0:3]
labels_D3_test=labels_D3[3:5]
labels_R_train=labels_R[0:3]
labels_R_test=labels_D1[3:5]'''


#Buliding Network
def build_model(input_img):


    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = Dropout(0.5)(conv4)

    deconv4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(pool4)
    uconv4 = concatenate([deconv4, conv3])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv2])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv1])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(192, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(uconv2)

    output_img = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(uconv2)
    output_img1 = Lambda(lambda output_img: output_img[:, :, :, 0])(output_img)
    output_img2 = Lambda(lambda output_img: output_img[:, :, :, 0])(output_img)
    output_img3 = Lambda(lambda output_img: output_img[:, :, :, 0])(output_img)

    output_img1 = Reshape((256, 256, 1))(output_img1)
    output_img2 = Reshape((256, 256, 1))(output_img2)
    output_img3 = Reshape((256, 256, 1))(output_img3)

    output_img11 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(output_img1)
    output_img22 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(output_img2)
    output_img33 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(output_img3)
    output_img = h.Add()([output_img11, output_img22, output_img33])

    return output_img1, output_img2, output_img3, output_img

input_img = Input(shape=(256,256,1))
labels_layer_D1=Input(shape=(256,256,1))
labels_layer_D2=Input(shape=(256,256,1))
labels_layer_D3=Input(shape=(256,256,1))
labels_layer_R=Input(shape=(256,256,1))
output_img1, output_img2, output_img3, output_img=build_model(input_img)


model = Model(inputs=[input_img, labels_layer_D1, labels_layer_D2, labels_layer_D3, labels_layer_R], outputs=[output_img1, output_img2, output_img3, output_img])

batch_size=4

#styles - style 3 and 4 best
#style1
'''l1_D=0.1*(k.sqrt(k.sum(k.square(h.Subtract()([output_img1,labels_layer_D1])))))+0.9*(k.sum(k.abs(h.Subtract()([output_img1,labels_layer_D1]))))
l2_D=0.1*(k.sqrt(k.sum(k.square(h.Subtract()([output_img2,labels_layer_D2])))))+0.9*(k.sum(k.abs(h.Subtract()([output_img2,labels_layer_D2]))))
l3_D=0.1*(k.sqrt(k.sum(k.square(h.Subtract()([output_img3,labels_layer_D3])))))+0.9*(k.sum(k.abs(h.Subtract()([output_img3,labels_layer_D3]))))
l_D=(l1_d+l2_D+l3_D)/batch_size
l_R=k.sqrt(k.sum(k.square(h.Subtract()([output_img3,labels_layer_D3]))))/batch_size
loss=l_D+0.5*l_R'''

#style2
'''l1_D2=(k.abs(output_img1-labels_layer_D1))
l2_D2=(k.abs(output_img2-labels_layer_D2))
l3_D2=(k.abs(output_img3-labels_layer_D3))
l_D2=0.9*(l1_D2+l2_D2+l3_D2)

l1_D1=k.square(output_img1-labels_layer_D1)
l2_D1=k.square(output_img2-labels_layer_D2)
l3_D1=k.square(output_img3-labels_layer_D3)
l_D1=0.1*k.sqrt(l1_D1+l2_D1+l3_D1)

loss_D=(l_D1+l_D2)/batch_size
loss_R=k.sqrt(k.square(output_img-labels_layer_R))/batch_size
loss=loss_D+0.5*loss_R'''

#style3
loss1=0
loss2=0

for i in range (0, batch_size):
    l1=k.abs(output_img1[i]-labels_layer_D1[i])
    l2=k.abs(output_img2[i] - labels_layer_D2[i])
    l3=k.abs(output_img3[i] - labels_layer_D3[i])
    l=l1+l2+l3
    loss1=loss1+l
    loss2 = loss2+k.square(l)

loss1D=0.9*loss1/batch_size
loss2D=0.1*k.sqrt(loss2)/batch_size
lossD=loss1D+loss2D

loss3=0
for i in range (0, batch_size):
    l_R1 = k.abs(output_img[i] - labels_layer_R[i])
    loss3 = loss3 + k.square(l_R1)

lossR=k.sqrt(loss3)/batch_size
loss=lossD+0.5*lossR


'''#style4
loss2=0

for i in range (0, batch_size):
    l1=(output_img1[i]-labels_layer_D1[i])
    l2=(output_img2[i] - labels_layer_D2[i])
    l3=(output_img3[i] - labels_layer_D3[i])
    l=l1+l2+l3
    loss1=0.9*k.sum(k.abs(l))+0.1*k.sqrt(k.sum(k.square(l)))
    loss2 = loss2+loss1

lossD=loss2/batch_size

loss3=0
for i in range (0, batch_size):
    l_R1=k.sqrt(k.sum(k.abs(output_img[i]-labels_layer_D1[i])))
    loss3=loss3+l_R1

lossR=loss3/batch_size

loss=lossD+0.5*lossR'''

max_pixel = 1.0

metric=(10.0 * k.log((max_pixel ** 2) / (k.mean(k.square(output_img - labels_layer_R), axis=-1)))) / 2.303

'''metric_T=0
for i in range (0, batch_size):
    metric_R = (10.0 * k.log((max_pixel ** 2) / (k.mean(k.square(output_img[i] - labels_layer_R[i]), axis=-1)))) / 2.303
    metric_T=metric_T+metric_R

metric=metric_T/batch_size'''

model.add_loss(loss)
model.add_metric(metric)

#Compiling
model.compile(optimizer='adam')

#Fitting
model.fit([images, labels_D1, labels_D2, labels_D3, labels_R], epochs=10, batch_size=batch_size, shuffle=True)

# serialize model to JSON
model_json = model.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_new.h5")
print("Saved model to disk")
