#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:44:25 2020

@author: marcpozzo
"""

#In this script you extract tiny images from images make sure that Tiny_images exits
import pandas as pd
import cv2
import os
import Lenet_training_functions as fn
image_path='../../../../Pic_dataset/'
tiny_image_path='../../../../Tiny_images/'

Mat_path="../../Materiels/"
Images=pd.read_csv(Mat_path+"images.csv")

#set minimum number requirement
Minimum_Number_Class=int(input("A minimum size popoulation for each labels is required to well analaze dataset. at what amount do you want to set the size limit (100 adviced ?) "))
Images=fn.eliminate_small_categories(Images,Minimum_Number_Class)


#Create imagetteName feature (the name of tiny_images)
tiny_images_names_=[]
for i in range(len(Images)):
    tiny_images_names_.append(Images['filename'].iloc[i][:-4]+"_"+Images['classe'].iloc[i]+"_"+str(i)+".JPG")
Images['imagetteName']=tiny_images_names_
Images.to_csv(Mat_path+"Tiny_images.csv",index=False)
      
#Extraction of tiny images
#for i in range(len(Images)):
if os.path.exists(tiny_image_path)==True:
    print("Warning Tiny_images already exists ! If you want to generate tiny images, please delete first this folder and then retry." )
elif os.path.exists(tiny_image_path)==False:    
    os.mkdir(tiny_image_path)
    for i in range(len(Images)):
        xmin,ymin,xmax,ymax=Images[['xmin', 'ymin', 'xmax', 'ymax']].iloc[i]
        image_name=Images["filename"].iloc[i]
        image=cv2.imread(image_path+image_name)
        tiny_image=image[ymin:ymax,xmin:xmax]
        tiny_image_name=Images['imagetteName'].iloc[i]
        cv2.imwrite(tiny_image_path+tiny_image_name,tiny_image)