import cv2
import glob
import numpy as np
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input, Lambda, Reshape, Multiply
import keras.backend as k
from keras.models import Model
import os
import matplotlib.pyplot as plt
import pydicom
import keras.layers as h
import tensorflow as tf
import math

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
images = images.reshape((12, 256, 256, 1))
images = images.astype('float32') / 255

labels_D1=np.asarray(labels_D1)
labels_D1 = cv2.normalize(labels_D1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D1.astype(np.uint8)
labels_D1 = labels_D1.reshape((12, 256, 256, 1))
labels_D1 = labels_D1.astype('float32') / 255

labels_D2=np.asarray(labels_D2)
labels_D2 = cv2.normalize(labels_D2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D2.astype(np.uint8)
labels_D2 = labels_D2.reshape((12, 256, 256, 1))
labels_D2 = labels_D2.astype('float32') / 255

labels_D3=np.asarray(labels_D3)
labels_D3 = cv2.normalize(labels_D3, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
labels_D3.astype(np.uint8)
labels_D3 = labels_D3.reshape((12, 256, 256, 1))
labels_D3 = labels_D3.astype('float32') / 255

#Train-Test split
images_train=images[0:9]
images_test=images[9:12]
labels_D1_train=labels_D1[0:9]
labels_D1_test=labels_D1[9:12]
labels_D2_train=labels_D2[0:9]
labels_D2_test=labels_D1[9:12]
labels_D3_train=labels_D3[0:9]
labels_D3_test=labels_D3[9:12]

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

    '''output_img1 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(uconv2)
    output_img2 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(uconv2)
    output_img3 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(uconv2)'''

    output_img = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(uconv2)
    output_img1 = Lambda(lambda output_img: output_img[:, :, :, 0])(output_img)
    output_img2 = Lambda(lambda output_img: output_img[:, :, :, 0])(output_img)
    output_img3 = Lambda(lambda output_img: output_img[:, :, :, 0])(output_img)

    output_img1 = Reshape((256, 256, 1))(output_img1)
    output_img2 = Reshape((256, 256, 1))(output_img2)
    output_img3 = Reshape((256, 256, 1))(output_img3)

    return output_img1, output_img2, output_img3

input_img = Input(shape=(256,256,1))
labels_layer_D1=Input(shape=(256,256,1))
labels_layer_D2=Input(shape=(256,256,1))
labels_layer_D3=Input(shape=(256,256,1))
output_img1, output_img2, output_img3 =build_model(input_img)


model = Model(inputs=[input_img, labels_layer_D1, labels_layer_D2, labels_layer_D3], outputs=[output_img1, output_img2, output_img3])

batch_size=3
validation_split=0.2
train_size=math.ceil(batch_size*validation_split)

#styles - style 3 and 4 best
#style1
'''l1_D=0.1*(k.sqrt(k.sum(k.square(h.Subtract()([output_img1,labels_layer_D1])))))+0.9*(k.sum(k.abs(h.Subtract()([output_img1,labels_layer_D1]))))
l2_D=0.1*(k.sqrt(k.sum(k.square(h.Subtract()([output_img2,labels_layer_D2])))))+0.9*(k.sum(k.abs(h.Subtract()([output_img2,labels_layer_D2]))))
l3_D=0.1*(k.sqrt(k.sum(k.square(h.Subtract()([output_img3,labels_layer_D3])))))+0.9*(k.sum(k.abs(h.Subtract()([output_img3,labels_layer_D3]))))
loss=(l1_d+l2_D+l3_D)/batch_size'''

#style2
'''l1_D2=(k.abs(output_img1-labels_layer_D1))
l2_D2=(k.abs(output_img2-labels_layer_D2))
l3_D2=(k.abs(output_img3-labels_layer_D3))
l_D2=0.9*(l1_D2+l2_D2+l3_D2)

l1_D1=k.square(output_img1-labels_layer_D1)
l2_D1=k.square(output_img2-labels_layer_D2)
l3_D1=k.square(output_img3-labels_layer_D3)
l_D1=0.1*k.sqrt(l1_D1+l2_D1+l3_D1)

loss=(l_D1+l_D2)'''

#loss=loss_D+0.5*loss_R

#style3
loss1=0
loss2=0

for i in range (0, train_size):
    l1=k.abs(output_img1[i]-labels_layer_D1[i])
    l2=k.abs(output_img2[i] - labels_layer_D2[i])
    l3=k.abs(output_img3[i] - labels_layer_D3[i])
    l=l1+l2+l3
    loss1=loss1+l
    loss2 = loss2+k.square(l)

loss1D=0.9*loss1/train_size
loss2D=0.1*k.sqrt(loss2)/train_size
loss=loss1D+loss2D


'''#style4
loss2=0

for i in range (0, batch_size):
    l1=(output_img1[i]-labels_layer_D1[i])
    l2=(output_img2[i] - labels_layer_D2[i])
    l3=(output_img3[i] - labels_layer_D3[i])
    l=l1+l2+l3
    loss1=0.9*k.sum(k.abs(l))+0.1*k.sqrt(k.sum(k.square(l)))
    loss2 = loss2+loss1

loss=loss2/batch_size'''

#Metric PSNR

#max_pixel = 1.0
#metric=(10.0 * k.log((max_pixel ** 2) / (k.mean(k.square((output_img1 - labels_layer_D1)+(output_img2 - labels_layer_D2)+(output_img3 - labels_layer_D3)))))) / 2.303

'''metric_T1=0
for i in range (0, batch_size):
    metric_R1 = (10.0 * k.log((max_pixel ** 2) / (k.mean(k.square(output_img1[i] - labels_layer_D1[i]))))) / 2.303
    metric_T1=metric_T1+metric_R1

metric1=metric_T1/batch_size

metric_T2=0
for i in range (0, batch_size):
    metric_R2 = (10.0 * k.log((max_pixel ** 2) / (k.mean(k.square(output_img2[i] - labels_layer_D2[i]))))) / 2.303
    metric_T2=metric_T2+metric_R2

metric2=metric_T2/batch_size

metric_T3=0
for i in range (0, batch_size):
    metric_R3 = (10.0 * k.log((max_pixel ** 2) / (k.mean(k.square(output_img3[i] - labels_layer_D3[i]))))) / 2.303
    metric_T3=metric_T3+metric_R3

metric3=metric_T3/batch_size
metric=(metric1+metric2+metric3)/3'''

#Metric SSIM

k1=0.01
k2=0.03
max_val=255

c1 = (k1 * max_val)**2
c2 = (k2 * max_val)**2

a1=((2*(k.mean(output_img1))*(k.mean(labels_layer_D1))+c1)*((2*(k.mean(Multiply()([output_img1,labels_layer_D1])))-(k.mean(output_img1))*(k.mean(labels_layer_D1)))+c2))
b1=((k.square(k.mean(output_img1))+k.square(k.mean(labels_layer_D1))+c1)*(k.square(k.var(output_img1))+k.square(k.var(labels_layer_D1))+c2))

metric1=a1/b1

a2=((2*(k.mean(output_img2))*(k.mean(labels_layer_D2))+c1)*((2*(k.mean(Multiply()([output_img2,labels_layer_D2])))-(k.mean(output_img2))*(k.mean(labels_layer_D2)))+c2))
b2=((k.square(k.mean(output_img2))+k.square(k.mean(labels_layer_D2))+c1)*(k.square(k.var(output_img2))+k.square(k.var(labels_layer_D2))+c2))

metric2=a2/b2

a3=((2*(k.mean(output_img3))*(k.mean(labels_layer_D3))+c1)*((2*(k.mean(Multiply()([output_img3,labels_layer_D3])))-(k.mean(output_img3))*(k.mean(labels_layer_D3)))+c2))
b3=((k.square(k.mean(output_img3))+k.square(k.mean(labels_layer_D3))+c1)*(k.square(k.var(output_img3))+k.square(k.var(labels_layer_D3))+c2))

metric3=a3/b3
metric=(metric1+metric2+metric3)/3

model.add_loss(loss)
model.add_metric(metric, name='ssim')

#Compiling
model.compile(optimizer='adam')






#Fitting
hist=model.fit([images_train, labels_D1_train, labels_D2_train, labels_D3_train], epochs=2, batch_size=batch_size, shuffle=True, validation_split=validation_split )

# serialize model to JSON
model_json = model.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_new.h5")
print("Saved model to disk")

test_loss, test_ssim = model.evaluate([images_test, labels_D1_test, labels_D2_test, labels_D3_test])

test_loss="%.4f" % round(test_loss, 4)
test_ssim="%.4f" % round(test_ssim, 4)

print('test_loss:',test_loss,'-', 'test_ssim:',test_ssim)

#Plot Accuracy
f=plt.figure(1)
plt.plot(hist.history['ssim'])
plt.plot(hist.history['val_ssim'])
plt.title('Model SSIM')
plt.ylabel('SSIM')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
f.show()

#Plot Loss
g=plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')

plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
g.show()

outputs= model.predict([images_test, images_test, images_test, images_test])
outputs=np.asarray(outputs)
h=images_test.shape[0]

for i in range(0, h):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Original DRR')
    plt.imshow(images_test[i].reshape((256,256)))
    plt.subplot(2, 2, 2)
    plt.title('DRR Ribs')
    plt.imshow(outputs[0][i].reshape((256,256)))
    plt.subplot(2, 2, 3)
    plt.title('DRR Vascular')
    plt.imshow(outputs[1][i].reshape((256,256)))
    plt.subplot(2, 2, 4)
    plt.title('DRR Spine')
    plt.imshow(outputs[2][i].reshape((256,256)))
    plt.show()


#model = load_model('model_new.h5')
