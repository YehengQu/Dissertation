# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:39:10 2020

@author: Yeheng
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:08:23 2020

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


def to_class_no(y_hot_list):
    y_class_list = []
    
    n = len(y_hot_list)
    
    for i in range(n):
        
        out = np.argmax(y_hot_list[i])
        
        y_class_list.append(out)
        
    return y_class_list


def conf_matrix(Y_gt, Y_pred, num_classes = 3):
    
    total_pixels = 0
    kappa_sum = 0
    sudo_confusion_matrix = np.zeros((num_classes, num_classes))
   
#    if len(Y_pred.shape) == 3:
#        h,w,c = Y_pred.shape
#        Y_pred = np.reshape(Y_pred, (1,))
 
    n = len(Y_pred)
    
    for i in range(n):
        y_pred = Y_pred[i]
        y_gt = Y_gt[i]
        
        #y_pred_hotcode = hotcode(y_pred)
        #y_gt_hotcode = hotcode(y_gt)
        
        pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
        gt = np.reshape(y_gt, (y_gt.shape[0]*y_gt.shape[1], y_gt.shape[2]))
        
        pred = [i for i in pred]
        gt = [i for i in gt]
        
        pred = to_class_no(pred)
        gt = to_class_no(gt)
        
#        pred.tolist()
#        gt.tolist()

        gt = np.asarray(gt, dtype = 'int32')
        pred = np.asarray(pred, dtype = 'int32')

        conf_matrix = confusion_matrix(gt, pred, labels=[0,1,2])
        
        kappa = cohen_kappa_score(gt,pred, labels=[0,1,2])

        pixels = len(pred)
        total_pixels = total_pixels+pixels
        
        sudo_confusion_matrix = sudo_confusion_matrix + conf_matrix
        
        kappa_sum = kappa_sum + kappa

    final_confusion_matrix = sudo_confusion_matrix
    
    final_kappa = kappa_sum/n

    return final_confusion_matrix, final_kappa


pred=io.imread('C:/Users/yehen/OneDrive/Desktop/eye-in-the-sky-master/outputs_fcn_843/out13.jpg')
label=io.imread('C:/Users/yehen/OneDrive/Desktop/eye-in-the-sky-master/outputs_fcn_843/label13.jpg')
a=label[:,:,0]
a[a<50]=0
a[a>=50]=125
b=np.zeros_like(a)
c=np.zeros_like(a)
print(a)
label=np.zeros([256,256,3])
label[:,:,0]=b
label[:,:,1]=a
label[:,:,2]=c
plt.imshow(label)

e=label[:,:,1]

pred_list=[]
label_list=[]
pred_list.append(pred)
label_list.append(label)
confusion_matrix_train, kappa_train = conf_matrix(label_list, pred_list, num_classes = 3)
print('Confusion Matrix for training')
print(confusion_matrix_train)
print('Kappa Coeff for training without unclassified pixels')
print(kappa_train)



def acc_of_class(class_label, conf_matrix, num_classes = 3):
    
    numerator = conf_matrix[class_label, class_label]
    
    denorminator = 0
    
    for i in range(num_classes):
        denorminator = denorminator + conf_matrix[class_label, i]
        
    acc_of_class = numerator/denorminator
    
    return acc_of_class


# On training

# Find accuray of all the classes NOT considering the unclassified pixels

# for i in range(2):
#     acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 2)
#     print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Training')
#     print(acc_of_cl)

# # Find accuray of all the classes considering the unclassified pixels

# for i in range(3):
#     acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 3)
#     print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Training')
#     print(acc_of_cl)
    
    
def overall_acc(conf_matrix, include_unclassified_pixels = False):
    
    if include_unclassified_pixels:
        
        numerator = 0
        for i in range(3):
        
            numerator = numerator + conf_matrix[i,i]
        
        denominator = 0   
        for i in range(3):
            for j in range(3):
                
                denominator = denominator + conf_matrix[i,j]
                
        acc = numerator/denominator
        
        return acc
    
    else:
        
        numerator = 0
        for i in range(2):
        
            numerator = numerator + conf_matrix[i,i]
        
        denominator = 0   
        for i in range(2):
            for j in range(2):
            
                denominator = denominator + conf_matrix[i,j]
                
        acc = numerator/denominator
        
        return acc
    
print('Over all accuracy WITHOUT unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = True))