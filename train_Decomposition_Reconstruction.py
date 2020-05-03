import cv2
import glob
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input, Multiply
import keras.backend as k
from keras.models import Model
import os
import matplotlib.pyplot as plt
import pydicom
import keras.layers as h
from tensorflow_core.python.ops.image_ops_impl import ssim

images=[]

images.append(np.zeros((2, 2)))
d=[]
d.append([1,1])
d.append([1,1])
images.append(d)
images=np.asarray(images)
images = images.reshape((2, 2, 2, 1))

labels_R=[]
a=[]
b=[]
a.append([1,1])
a.append([1,1])
b.append([2,2])
b.append([2,2])
labels_R.append(a)
labels_R.append(b)

labels_R=np.asarray(labels_R)
labels_R=labels_R.reshape((2, 2, 2, 1))



#Buliding Network
def build_model(input_img):

    output_img = MaxPooling2D((1, 1))(input_img)


    return output_img

input_img = Input(shape=(2,2,1))
labels_layer_R=Input(shape=(2,2,1))
output_img = build_model(input_img)


model = Model(inputs=[input_img, labels_layer_R], outputs=[output_img])

loss=(k.abs(output_img-labels_layer_R))

k1=0.01
k2=0.03
max_val=1

c1 = (k1 * max_val)**2
c2 = (k2 * max_val)**2

metric=((2*k.abs(k.mean(output_img))*k.abs(k.mean(labels_layer_R))+c1)*(2*k.abs((k.mean(Multiply()([output_img,labels_layer_R])))-k.abs(k.mean(output_img))*k.abs(k.mean(labels_layer_R)))+c2))/((k.square(k.mean(output_img))+k.square(k.mean(labels_layer_R))+c1)*(k.var(output_img)+k.var(labels_layer_R)+c2))
#metric=(2*k.abs(k.mean(output_img))*k.abs(k.mean(labels_layer_R)))
#metric=(2*k.abs((k.mean(Multiply()([output_img,labels_layer_R])))-k.abs(k.mean(output_img))*k.abs(k.mean(labels_layer_R))))

model.add_loss(loss)
model.add_metric(metric)


#Compiling
model.compile(optimizer='adam')

#Fitting
model.fit([images, labels_R], epochs=2, batch_size=2, shuffle=True)

