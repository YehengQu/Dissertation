# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:17:34 2020

@author: Yeheng
"""

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
config.gpu_options.per_process_gpu_memory_fraction = 0.5 
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

del(trainy,testy)
VGG_Weights_path ='vgg16_weights.h5'
def FCN8( nClasses=3 ,  input_height=256, input_width=256):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 

    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3
    
    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)

    #x = Flatten(name='flatten')(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data
    
    #x = Dense(4096, activation='relu', name='fc2')(x)
    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)   
    # assuming that the input_height = input_width = 224 as in VGG data
    
    #x = Dense(1000 , activation='softmax', name='predictions')(x)
    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data
    
    
    vgg  = Model(img_input, pool5)
    vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8
    
    n = 2048
    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)
    
    
    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(4,4) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(pool411)
    
    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
        
    o = Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o = (Activation('softmax'))(o)
    
    model = Model(img_input, o)
    
    model.compile(optimizer = Adam(lr = 0.0001), loss="categorical_crossentropy", metrics = ['accuracy'])
    
    model.summary()
    return model




 


# def fcn(input_shape = (None,None,3),batch_size=batch_size,classes=3):
    
#     img_input = Input(input_shape)

#     # Block 1
#     x = Conv2D(64, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block1_conv1')(img_input)
#     x = Conv2D(64, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block1_conv2')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
 
#     # Block 2
#     x = Conv2D(128, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block2_conv1')(x)
#     x = Conv2D(128, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block2_conv2')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
 
#     # Block 3
#     x = Conv2D(256, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block3_conv1')(x)
#     x = Conv2D(256, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block3_conv2')(x)
#     x = Conv2D(256, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block3_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
 
#     # Block 4
#     x = Conv2D(512, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block4_conv1')(x)
#     x = Conv2D(512, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block4_conv2')(x)
#     x = Conv2D(512, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block4_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
 
#     # Block 5
#     x = Conv2D(512, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block5_conv1')(x)
#     x = Conv2D(512, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block5_conv2')(x)
#     x = Conv2D(512, (3, 3),
#                       activation='relu',
#                       padding='same',
#                       name='block5_conv3')(x)
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
 
#     # FCN Classification block
#     x = Conv2D(2048,(7, 7),strides=(1,1),activation='relu',name='fc1',padding='valid')(x)
#     x = Conv2D(2048,(1, 1),strides=(1,1),activation='relu',name='fc2',padding='valid')(x)
#         # x = Conv2D(classes,(1, 1),strides=(1,1),activation='softmax',name='predictions',padding='valid')(x)

   
#     # dconv3_shape = tf.stack([img_input[0], img_input[1], img_input[2], classes])
#     # upsample_1 = upsample_layer(x, dconv3_shape, classes, 'upsample_1', 32)

#     # skip_1 = skip_layer_connection(pool4, 'skip_1', 512, stddev=0.00001)
#     # upsample_2 = upsample_layer(skip_1, dconv3_shape, classes, 'upsample_2', 16)

#     # skip_2 = skip_layer_connection(pool3, 'skip_2', 256, stddev=0.0001)
#     # upsample_3 = upsample_layer(skip_2, dconv3_shape, classes, 'upsample_3', 8)

#     # logit = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
        
#     # Create model.
#     # encoder_model = models.Model(img_input, x, name='vgg16_fcn')
    
  
#     x=UpSampling2D((2, 2))(x)

#     x=Deconvolution2D(512, 3, 3, output_shape=[batch_size, 3, 3, 512],
#                               border_mode='same', name='deconv5_1')(x)
#     x=Deconvolution2D(512, 3, 3, output_shape=[batch_size, 3, 3, 512],
#                               border_mode='same', name='deconv5_2')(x)
#     x=Deconvolution2D(512, 3, 3, output_shape=[batch_size, 3, 3, 512],
#                               border_mode='same', name='deconv5_3')(x)
    
#     x=UpSampling2D((2, 2))(x)

#     x=Deconvolution2D(512, 3, 3, output_shape=[batch_size, 3, 3, 512],
#                               border_mode='same', name='deconv4_1')(x)
#     x=Deconvolution2D(512, 3, 3, output_shape=[batch_size, 3, 3, 512],
#                               border_mode='same', name='deconv4_2')(x)
#     x=Deconvolution2D(256, 3, 3, (1, 256, output_size, output_size),
#                               border_mode='same', name='deconv4_3')(x)
    
#     x=UpSampling2D((2, 2))(x)

#     x=Deconvolution2D(256, 3, 3, (1, 256, output_size, output_size),
#                               border_mode='same', name='deconv3_1')(x)
#     x=Deconvolution2D(256, 3, 3, (1, 256, output_size, output_size),
#                               border_mode='same', name='deconv3_2')(x)
#     x=Deconvolution2D(128, 3, 3, (1, 128, output_size, output_size),
#                               border_mode='same', name='deconv3_3')(x)
    
#     x=UpSampling2D((2, 2))(x)

#     x=Deconvolution2D(128, 3, 3, (1, 128, output_size, output_size),
#                               border_mode='same', name='deconv2_1')(x)
#     x=Deconvolution2D(64, 3, 3, (1, 64, output_size, output_size),
#                               border_mode='same', name='deconv2_2')(x)
    
#     x=UpSampling2D((2, 2))(x)

#     x=Deconvolution2D(64, 3, 3, (1, 64, output_size, output_size),
#                               border_mode='same', name='deconv1_1')(x)
#     x=Deconvolution2D(64, 3, 3, (1, 64, output_size, output_size),
#                               border_mode='same', name='deconv1_2')(x)
#     x=Convolution2D(classes, 1, 1, activation='relu', name='output')(x)
    
    
#     model = models.Model(img_input, x, name='vgg16_fcn')

    # model.compile(optimizer = Adam(lr = 0.0001), loss="categorical_crossentropy", metrics = ['accuracy'])
    
    # model.summary()
 
#     return model
# def vgg_fcn(shape = (None,None,3)):
#     print("loading model...........................")
#     # Left side of the U-Net
#     inputs = Input(shape)
# #    in_shape = inputs.shape
# #    print(in_shape)

#     model = Sequential()
#     model.add(ZeroPadding2D((1,1)),batch_input_shape=inputs))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(128, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(128, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))

#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1,1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2,2), strides=(2,2)))
    
#     model.add(Convolution2D(4096, 7, 7, activation='relu',
#                             name='fc6'))
#     model.add(Dropout(0.5))
#     model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
#     model.add(Dropout(0.5))
    
#     output_size = 2 * model.layers[-1].output_shape[2]
#     model.add(UpSampling2D((2, 2)))
#     output_size = model.layers[-1].output_shape[2]
#     model.add(Deconvolution2D(512, 3, 3, (1, 512, output_size, output_size),
#                               border_mode='same', name='deconv5_1'))
#     model.add(Deconvolution2D(512, 3, 3, (1, 512, output_size, output_size),
#                               border_mode='same', name='deconv5_2'))
#     model.add(Deconvolution2D(512, 3, 3, (1, 512, output_size, output_size),
#                               border_mode='same', name='deconv5_3'))
    
#     model.add(UpSampling2D((2, 2)))
#     output_size = model.layers[-1].output_shape[2]
#     model.add(Deconvolution2D(512, 3, 3, (1, 512, output_size, output_size),
#                               border_mode='same', name='deconv4_1'))
#     model.add(Deconvolution2D(512, 3, 3, (1, 512, output_size, output_size),
#                               border_mode='same', name='deconv4_2'))
#     model.add(Deconvolution2D(256, 3, 3, (1, 256, output_size, output_size),
#                               border_mode='same', name='deconv4_3'))
    
#     model.add(UpSampling2D((2, 2)))
#     output_size = model.layers[-1].output_shape[2]
#     model.add(Deconvolution2D(256, 3, 3, (1, 256, output_size, output_size),
#                               border_mode='same', name='deconv3_1'))
#     model.add(Deconvolution2D(256, 3, 3, (1, 256, output_size, output_size),
#                               border_mode='same', name='deconv3_2'))
#     model.add(Deconvolution2D(128, 3, 3, (1, 128, output_size, output_size),
#                               border_mode='same', name='deconv3_3'))
    
#     model.add(UpSampling2D((2, 2)))
#     output_size = model.layers[-1].output_shape[2]
#     model.add(Deconvolution2D(128, 3, 3, (1, 128, output_size, output_size),
#                               border_mode='same', name='deconv2_1'))
#     model.add(Deconvolution2D(64, 3, 3, (1, 64, output_size, output_size),
#                               border_mode='same', name='deconv2_2'))
    
#     model.add(UpSampling2D((2, 2)))
#     output_size = model.layers[-1].output_shape[2]
#     model.add(Deconvolution2D(64, 3, 3, (1, 64, output_size, output_size),
#                               border_mode='same', name='deconv1_1'))
#     model.add(Deconvolution2D(64, 3, 3, (1, 64, output_size, output_size),
#                               border_mode='same', name='deconv1_2'))
#     model.add(Convolution2D(classes, 1, 1, activation='relu', name='output'))
    
model = FCN8()

history = model.fit(trainx, trainy_hot, epochs=20, validation_data = (testx, testy_hot),batch_size=1, verbose=1)
model.save("model_fcn_1182.h5")

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

