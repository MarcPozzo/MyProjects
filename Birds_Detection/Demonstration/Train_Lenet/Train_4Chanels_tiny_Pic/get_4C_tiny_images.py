#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:43:44 2020

@author: marcpozzo
"""

#Import libraries
import pandas as pd
import functions_4C as fn
from sklearn.model_selection import train_test_split
import gc
from numpy import save
Mat_path="../../../Materiels/"

image_path='../../../../../Pic_dataset/'
tiny_image_path='../../../../../4C_tiny_picture/'




print("The 4 chanels correspond to the difference between an image and the image just before. In sense add this 4TH chanels could be considered as to add a temporel chanels.")
print("the difference between the 2 pictures is made in grey colors.")
print("If you want to apply the conversion of images to gray and the difference that follows type HSV.")
print("If you want to apply filters before converting to gray, type GBR.")
color_space_diff=input("Please type HSV or GBR ")
print("Type train or test wethever you want to rec picture with a fourth chanel for train or test sample")
print( "I advise you to beguin with small proportion to be sure that your computer has enough memory get pictures")
Sample=input(" Please type train or test " )
test_size = float(input("Type a number between 0.1 and 0.5 to indicate the proportion of data you want put in test sample : "))




Images=pd.read_csv(Mat_path+"Images.csv")

#set minimum number requirement
Minimum_Number_Class=int(input("A minimum size popoulation for each labels is required to well analaze dataset. at what amount do you want to set the size limit (100 adviced ?) "))
Images=fn.eliminate_small_categories(Images,Minimum_Number_Class)




base_train,base_test= train_test_split(Images,stratify=Images["classe"], test_size=test_size,random_state=42)
if Sample=="train":
    base=base_train
    del Images,base_test
elif Sample=="test":
    base=base_test
    del Images,base_train 
else:
    print("The 4th pictures weren't generated because you didin't type train or test (in lower case). Pleas try again")
gc.collect()




X=fn.get_X(base[:2],image_path,color_space_diff)
Y=fn.get_Y(base[:2])
save(tiny_image_path+'imagettes4C_'+color_space_diff+'_'+Sample +'.npy', X)
save(tiny_image_path+'labels4C_'+color_space_diff+'_'+Sample +'.npy', Y)
gc.collect()