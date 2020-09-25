# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:04:19 2020

@author: Yeheng
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:28:53 2020

@author: Yeheng
"""

import PIL
from PIL import Image
from keras import models
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
# from scipy.misc import imresize
import numpy as np
import math
import glob
import cv2
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
# %matplotlib inline
import os
from keras.utils.np_utils import to_categorical

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'



def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc


def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def tf_mean_iou(y_true, y_pred, num_classes=8):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)


mean_iou = as_keras_metric(tf_mean_iou)


# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# List of file names of actual Satellite images for traininig 
filelist_trainx = sorted(glob.glob('C:/Users/yehen/OneDrive/Desktop/train1182/*.jpg'), key=numericalSort)
# List of file names of classified images for traininig 
filelist_trainy = sorted(glob.glob('C:/Users/yehen/OneDrive/Desktop/train1182/label*.png'), key=numericalSort)



# Resizing the image to nearest dimensions multipls of 'stride'

def resize(img, stride, n_h, n_w):
    #h,l,_ = img.shape
    ne_h = (n_h*stride) + stride
    ne_w = (n_w*stride) + stride
    
    img_resized = numpy.array(Image.fromarray(arr).resize())(img, (ne_h,ne_w))
    return img_resized

# Padding at the bottem and at the left of images to be able to crop them into 128*128 images for training

def padding(img, w, h, c, crop_size, stride, n_h, n_w):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    #img_pad[:h, :w,:] = img
    #img_pad = img_pad+img
    if len(img.shape)==2:
        img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd)], mode='constant')
    else:
        img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    
    return img_pad
    


# Adding pixels to make the image with shape in multiples of stride

def add_pixals(img, h, w, c, n_h, n_w, crop_size, stride):
        
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
        
    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]
            
    return img_add    



# Adding pixels to make the image with shape in multiples of stride

def add_pixals(img, h, w, c, n_h, n_w, crop_size, stride):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
    
    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]
    
    return img_add    


# Slicing the image into crop_size*crop_size crops with a stride of crop_size/2 and makking list out of them

def crops(a, crop_size = 256):
    
    stride = int(crop_size/2)
    # stride = 128

    croped_images = []
    if len(a.shape)==2:
        c=1
        h,w=a.shape
    else:   
        h, w, c = a.shape
        
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    # Padding using the padding function we wrote
    a = padding(a, w, h, c, crop_size, stride, n_h, n_w)
    
    # Resizing as required
    ##a = resize(a, stride, n_h, n_w)
    
    # Adding pixals as required
    #a = add_pixals(a, h, w, c, n_h, n_w, crop_size, stride)
    
    # Slicing the image into 128*128 crops with a stride of 64
    if len(a.shape)==2:
        for i in range(n_h-1):
            for j in range(n_w-1):
                crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size)]
                croped_images.append(crop_x)
    else:
        for i in range(n_h-1):
            for j in range(n_w-1):
                crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
                croped_images.append(crop_x)
    return croped_images


######################################################################################以下的正常
# Reading, padding, cropping and making array of all the cropped images of all the trainig sat images
trainx_list = []

for fname in filelist_trainx[:7]:
    
    image=io.imread(fname)
    # Padding as required and cropping
    crops_list = crops(image)
    #print(len(crops_list))
    trainx_list = trainx_list + crops_list
    
# Array of all the cropped Training sat Images    
trainx = np.asarray(trainx_list,dtype="float16")


# Reading, padding, cropping and making array of all the cropped images of all the trainig gt images
trainy_list = []

for fname in filelist_trainy[:7]:
    

    image=io.imread(fname)
    
    # Padding as required and cropping
    crops_list =crops(image)
    
    trainy_list = trainy_list + crops_list
    
# Array of all the cropped Training gt Images    
trainy = np.asarray(trainy_list,dtype="int64")
trainy = trainy[:,:,:,:3]
for i in range(trainy.shape[0]):
    a=trainy[i,:,:,0]
    a[a<50]=0
    a[a>=50]=125
    b=np.zeros_like(a)
    c=np.zeros_like(a)
    d=np.zeros([256,256,3])
    d[:,:,0]=b
    d[:,:,1]=a
    d[:,:,2]=c
    trainy[i,:,:,:]=d
    


############################################################################
# Reading, padding, cropping and making array of all the cropped images of all the testing sat images
testx_list = []

image=io.imread(filelist_trainx[7])
    
# Padding as required and cropping
crops_list = crops(image)
    
testx_list = testx_list + crops_list
    
# Array of all the cropped Testing sat Images  
testx = np.asarray(testx_list,dtype="float16")


# Reading, padding, cropping and making array of all the cropped images of all the testing sat images
testy_list = []

#for fname in filelist_trainx[13]:

image=io.imread(filelist_trainy[7])

# Padding as required and cropping
crops_list = crops(image)
    
testy_list = testy_list + crops_list
if len(testx_list) != len(testy_list):
    testy_list=testy_list[:testx_list]
    
# Array of all the cropped Testing sat Images 
testy = np.asarray(testy_list,dtype="int64") 
# testy = np.asarray(testy_list,dtype="float16").reshape(-1,256,256,1)
testy = testy[:,:,:,:3]

for i in range(testy.shape[0]):
    a=testy[i,:,:,0]
    a[a<50]=0
    a[a>=50]=125
    b=np.zeros_like(a)
    c=np.zeros_like(a)
    d=np.zeros([256,256,3])
    d[:,:,0]=b
    d[:,:,1]=a
    d[:,:,2]=c
    testy[i,:,:,:]=d
    

color_dict = {0: (0, 0, 0),
              1:(0,125,0),
              2:(255,255,255)}

# palette = [[0], [128], [255]]

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    # print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

# def mask_to_onehot(mask, palette):
#     """
#     Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
#     hot encoding vector, C is usually 1 or 3, and K is the number of class.
#     """
#     semantic_map = []
#     for colour in palette:
#         equality = np.equal(mask, colour)
#         class_map = np.all(equality, axis=-1)
#         semantic_map.append(class_map)
#     semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
#     return semantic_map
# def onehot_to_mask(mask, palette):
#     """
#     Converts a mask (H, W, K) to (H, W, C)
#     """
#     x = np.argmax(mask, axis=-1)
#     colour_codes = np.array(palette)
#     x = np.uint8(colour_codes[x.astype(np.uint8)])
#     return x
# Convert trainy and testy into one hot encode

# trainy_hot = []

# #    里面值的顺序不是固定的，可以按自己的要求来
# # 注意：灰度图的话要确保 gt的 shape = [H, W, 1]，该函数实在最后的通道维上进行映射
# # 如果加载后的gt的 shape = [H, W]，则需要进行通道的扩维
# for i in range(trainy.shape[0]):
#     hot_img=mask_to_onehot(trainy[i,:,:],palette)
#     trainy_hot.append(hot_img)
# trainy_hot = np.asarray(trainy_hot,dtype="float16")


# testy_hot = []

# for i in range(testy.shape[0]):
    
#     hot_img=mask_to_onehot(testy[i],palette)
#     testy_hot.append(hot_img)
# testy_hot.append(hot_img)

# testy_hot = np.asarray(testy_hot,dtype="float16")

# if testy_hot.shape[0]!= testx.shape[0]:
#     testy_hot=testy_hot[:testx.shape[0],:,:,:]

# del(testy,trainy)

trainy_hot = []

for i in range(trainy.shape[0]):
    
    hot_img = rgb_to_onehot(trainy[i], color_dict)
    
    trainy_hot.append(hot_img)
    
trainy_hot = np.asarray(trainy_hot)

testy_hot = []
for i in range(testy.shape[0]):
    
    hot_img = rgb_to_onehot(testy[i], color_dict)
    
    testy_hot.append(hot_img)
    
testy_hot = np.asarray(testy_hot)

def unet(shape = (None,None,3)):
    print("loading model...........................")
    # Left side of the U-Net
    inputs = Input(shape)
#    in_shape = inputs.shape
#    print(in_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottom of the U-Net
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Upsampling Starts, right side of the U-Net
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output layer of the U-Net with a softmax activation
    conv10 = Conv2D(3, 1, activation = 'softmax')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.summary()
    
    #filelist_modelweights = sorted(glob.glob('*.h5'), key=numericalSort)
    
    #if 'model_nocropping.h5' in filelist_modelweights:
      #   model.load_weights('model_nocropping.h5')
    ##model.load_weights("model_onehot.h5")
    return model

model = unet()


history = model.fit(trainx, trainy_hot, epochs=20, validation_data = (testx, testy_hot),batch_size=5, verbose=1)
model.save("model_unet_1182.h5")

# model.load("model_onehot.h5")
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_plot.png')
plt.show()
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('loss_plot.png')
plt.show()
plt.close()

#epochs = 20
#for e in range(epochs):
#        print("epoch %d" % e)
#        #for X_train, Y_train in zip(x_train, y_train): # these are chunks of ~10k pictures
#        h,w,c = x_train.shape
#        X_train = np.reshape(x_train,(1,h,w,c))
#        h,w,c = y_train.shape
#        Y_train = np.reshape(y_train,(1,h,w,c))
#        model.fit(X_train, Y_train, batch_size=1, nb_epoch=1)
 	
#        model.save("model_nocropping.h5")        
#print(X_train.shape, Y_train.shape)


#model.save("model_nocropping.h5")

#epochs = 10
#for e in range(epochs):
#	print("epoch %d" % e)
#	for X_train, Y_train in zip(x_train, y_train): # these are chunks of ~10k pictures
#		h,w,c = X_train.shape
#		X_train = np.reshape(X_train,(1,h,w,c))
#		h,w,c = Y_train.shape
#		Y_train = np.reshape(Y_train,(1,h,w,c))
#		model.fit(X_train, Y_train, batch_size=1, nb_epoch=1)
        #print(X_train.shape, Y_train.shape)


#model.save("model_nocropping.h5")

#accuracy = model.evaluate(x=x_test,y=y_test,batch_size=16)
#print("Accuracy: ",accuracy[1])

