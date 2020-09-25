# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:33:31 2020

@author: Yeheng
"""

import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
# from scipy.misc import imresize
import numpy as np
import glob
import cv2
import os
import math
from sklearn.metrics import confusion_matrix, cohen_kappa_score
# from main_VGG import UNet
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
# from scipy.misc import imsave
from keras import backend as K
from iou import iou
#%matplotlib inline
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

model = load_model('model_unet_1182.h5')
print("model load successfully!")


# List of file names of actual Satellite images for traininig 
filelist_trainx = sorted(glob.glob('C:/Users/yehen/OneDrive/Desktop/train1182/*.jpg'), key=numericalSort)
# List of file names of classified images for traininig 
filelist_trainy = sorted(glob.glob('C:/Users/yehen/OneDrive/Desktop/train1182/label*.png'), key=numericalSort)

# List of file names of actual Satellite images for testing 
filelist_testx = sorted(glob.glob('C:/Users/yehen/OneDrive/Desktop/validation1182/8.jpg'), key=numericalSort)

filelist_testy = sorted(glob.glob('C:/Users/yehen/OneDrive/Desktop/validation1182/label8.png'), key=numericalSort)
# Not useful, messes up with the 4 dimentions of sat images

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

def add_pixals(img, h, w, c, n_h, n_w, crop_size, stride):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img=img.reshape(h,w,c)
    
    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
    
    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]
    
    return img_add  
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


xtest_list1 = []
ytest_list1 = []
image_list = []
for fname in filelist_testx:
    
    # Reading the image
    # tif = TIFF.open(fname)
    # image = tif.read_image()
    image=io.imread(fname)
    crop_size = 256
    
    stride = int(crop_size/2)
    
    if len(image.shape)==2:
        c=1
        h,w=image.shape
    else:   
        h, w, c = image .shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride)
              )
    image_list=crops(image)
    # image_list.append(image)
    xtest_list1=xtest_list1+ image_list
    # image = add_pixals(image, h, w, c, n_h, n_w, crop_size, stride)
    
    # xtest_list1.append(image)

for fname in filelist_testy:
    
    # Reading the image
    # tif = TIFF.open(fname)
    # image = tif.read_image()
    label=io.imread(fname)
    crop_size = 256
    
    stride = int(crop_size/2)
    
    if len(label.shape)==2:
        c=1
        h,w=label.shape
    else:   
        h, w, c = label.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride)
              )
    label_list=crops(label)
    # image_list.append(image)
    ytest_list1=ytest_list1+ label_list
    
for i_ in range(len(ytest_list1)):  
    item = ytest_list1[i_]
    imx2 = Image.fromarray(item[:,:,:3])
    imx2.save("test_outputs_1182/label"+str(i_+1)+".jpg")
    
    
def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

color_dict = {0: (0, 0, 0),
              1:(0,125,0),
              2:(255,255,255)}


print("prepare to preict!")
y_list=[]
for i_ in range(len(xtest_list1)):
    
    item = xtest_list1[i_]
    
    h,w,c = item.shape
    
    item = np.reshape(item,(1,h,w,c))
    
    y_pred_test_img = model.predict(item,batch_size=5)
    
    print(i_)
    
    ba,h,w,c = y_pred_test_img.shape
    
    y_pred_test_img = np.reshape(y_pred_test_img,(h,w,c))
    
    img = y_pred_test_img
    h, w, c = img.shape
        
    for i in range(h):
        for j in range(w):
                
            argmax_index = np.argmax(img[i,j])
                
            sudo_onehot_arr = np.zeros((3))
                
            sudo_onehot_arr[argmax_index] = 1
                
            onehot_encode = sudo_onehot_arr
                
            img[i,j,:] = onehot_encode
    
    y_pred_test_img = onehot_to_rgb(img, color_dict)

    # # tif = TIFF.open(filelist_testx[i_])
    # # image2 = tif.read_image()
    # image2=io.imread(filelist_testx[i_])
    # h,w,c = image2.shape
    
    # y_pred_test_img = y_pred_test_img[:h, :w, :]
    imx = Image.fromarray(np.uint8(y_pred_test_img))
    imx1 = Image.fromarray(item.reshape(256,256,3))
    imx1.save("test_outputs_1182/yunatu"+str(i_+1)+".jpg")
    imx.save("test_outputs_1182/out"+str(i_+1)+".jpg")
    y_list.append(y_pred_test_img)

y_list=np.asarray( y_list,dtype="float16")

def class_prediction_to_image(im, PredictedTiles, size):

    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

     
    nTiles_height = h//size
    nTiles_width = w//size
    #TileTensor = np.zeros((nTiles_height*nTiles_width, size,size,d))
    TileImage = np.zeros(im.shape)
    B=0
    for y in range(0, nTiles_height):
        for x in range(0, nTiles_width):
            x1 = np.int32(x * size)
            y1 = np.int32(y * size)
            x2 = np.int32(x1 + size)
            y2 = np.int32(y1 + size)
            #TileTensor[B,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
            TileImage[y1:y2,x1:x2] = PredictedTiles[B,:,:,:]
            B+=1

    return TileImage

im3D=io.imread('C:/Users/yehen/OneDrive/Desktop/validation/9.jpg')
whole_img= class_prediction_to_image(im3D,y_list,256)
imx = Image.fromarray(np.uint8(whole_img))

imx.save("test_outputs/yuantu7.jpg")


