#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:44:25 2020

@author: marcpozzo
"""

#In this script you extract tiny images from images make sure that Tiny_images exits
import pandas as pd
import cv2
image_path='../../../../Pic_dataset/'
tiny_image_path='../../../../Tiny_images/'

Mat_path="../../Materiels/"
Images=pd.read_csv(Mat_path+"images.csv")




#Extraction of tiny images
for i in range(len(Images)):
    xmin,ymin,xmax,ymax=Images[['xmin', 'ymin', 'xmax', 'ymax']].iloc[i]
    image_name=Images["filename"].iloc[i]
    image=cv2.imread(image_path+image_name)
    tiny_image=image[ymin:ymax,xmin:xmax]
    tiny_image_name=Images['imagetteName'].iloc[i]
    cv2.imwrite(tiny_image_path+tiny_image_name,tiny_image)