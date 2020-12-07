#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:07:35 2020

@author: marcpozzo
"""

#This script gathers the functions used in the other script of this folder

#Import libraries
import cv2
import pandas as pd
from keras.layers import Dropout
import numpy as np
import gc
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D,Flatten
from keras.layers.convolutional import Conv2D

#Parameters to fixe
mat_path="../../../Materiels/"
fichierClasses= mat_path+"Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)



#Lenet architecture
def nn(dropout_rate=0.2):    
    
    lenet = Sequential()

    conv_1 = Conv2D(filters = 30,                     # Number of filters
                kernel_size = (5, 5),            # Dimensions of kernel
                padding = 'valid',               # Mode de Dépassement
                input_shape = (28, 28, 4),       # Dimension of entry here we have a 4 dimension images
                activation = 'relu')             # activation Function

    max_pool_1 = MaxPooling2D(pool_size = (2, 2))

    conv_2 = Conv2D(filters = 16,                    
                kernel_size = (3, 3),          
                padding = 'valid',             
                activation = 'relu')

    max_pool_2 = MaxPooling2D(pool_size = (2, 2))

    flatten = Flatten()

    dropout = Dropout(rate = dropout_rate)

    dense_1 = Dense(units = 128,
                activation = 'relu')

    dense_2 = Dense(units = 6,
                activation = 'softmax')

    lenet.add(conv_1)
    lenet.add(max_pool_1)
    lenet.add(conv_2)
    lenet.add(max_pool_2)
    lenet.add(dropout)
    lenet.add(flatten)
    lenet.add(dense_1)
    lenet.add(dense_2)



    
    
    
    lenet.compile(loss='sparse_categorical_crossentropy',  # fonction de perte
              optimizer='adam',                 # algorithme de descente de gradient
              metrics=['accuracy'])             # métrique d'évaluation
    
    return lenet




#Get labels for each pictures
def get_Y(base):

    Y = ImageDataGenerator().flow_from_dataframe(dataframe=base,directory="../../../../../Pic_dataset",y_col = "classe").classes

    
    return Y

#Get Pictures with an added 4th Chanels corresponding to the difference with the previous image
def get_4C_Pic(base,tiny_image_path,color_space_diff):
    
   
    #Gather tiny images in a list
    test_path=tiny_image_path+"Images_test/"
    tpicture_names_=list(base["imagetteName"].unique()) #tiny picture names
    path_timage_test_=[test_path+name for name in tpicture_names_]
    
    #Gather 3c picture in a list
    timages_3C_=[] #tiny images with 3 chanels
    for image_path in path_timage_test_:
        img=cv2.imread(image_path)
        img=cv2.resize(img,(28,28))
        timages_3C_.append(img)

    #Add 4th chanel depending on the difference pixel by pixel between this image and the previous one. 
    chanel_4_=make_image_difference(base,tiny_image_path,color_space_diff) #Chanels obtain with the difference of images
   
  
    #Concatenate 4th Chanel with the other 3 Chanels in a single array
    timages_4C_=list(map(add_chanel,timages_3C_ , chanel_4_))
    del timages_3C_
    gc.collect()
    X=np.array(timages_4C_)
    
    
    del timages_4C_
    gc.collect()

    
 
    
    return X

#Convert the method of writing a picture (HSV,GBR,RGB) to another
def convert_color(image,changement):
    if changement=="HSV":
        image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if changement=="BGRGRAY":
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if changement=="BGR":
        image=cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
    return image


#Add a 4th chanel to 3 exesting chanels from the pictures
def add_chanel(image,ar_in_liste_diff):

    b_channel, g_channel, r_channel = cv2.split(image)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, ar_in_liste_diff))
    
    return img_BGRA




# substract images and return the difference array 
def make_image_difference(base,tiny_image_path,color_space_diff="BGR"):

    Diff_4C_=[] #list gatehring the 4th chanel for every tiny images


    
    #For all images in the loop get the HSV and GBR diff
    test_path=tiny_image_path+"Images_test/"
    ref_path=tiny_image_path+"Images_ref/"
    for i in range(len(base)):
        #Open image test containing the birds and the previous one (image_ref)
        name_tpic=base["imagetteName"].iloc[i]
        name_test=test_path+name_tpic
        name_ref=ref_path+name_tpic
        imageA=cv2.imread(name_test)
        imageB=cv2.imread(name_ref)
        
        #difference for 3 chanels (BGR) and then convert to GRAY scale
        if color_space_diff=="BGR":
            BGR_Diff = cv2.absdiff(imageA, imageB)
            BGR_Diff=convert_color(BGR_Diff,"BGRGRAY")
            BGR_Diff=cv2.resize(BGR_Diff,(28,28))
            Diff_4C_.append(BGR_Diff)


        #Make the difference in HSV method and teg
        elif color_space_diff=="HSV":
            imgAHSV=convert_color(imageA,"HSV")
            imgBHSV=convert_color(imageB,"HSV")
            HSV = cv2.absdiff(imgAHSV, imgBHSV)
            HSV_Diff = convert_color(HSV,"BGRGRAY")
            HSV_Diff=cv2.resize(HSV_Diff,(28,28))
            Diff_4C_.append(HSV_Diff)
        
        else:
            print("Warning you miss type the name of color space difference")
    return Diff_4C_








