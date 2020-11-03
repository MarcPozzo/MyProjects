#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:30:46 2020

@author: marcpozzo
"""
#This script add a 4th chanel to the 3 chanels pictures. Tis 4th chanel corresponded to the difference pixel by pixel in Hsv between each pictures containing birds and the previous one.



#Import libraries
import pandas as pd
import functions_4C as fn
import time
from sklearn.model_selection import train_test_split
import gc
from numpy import save
Mat_path="../../../Materiels/"

image_path='../../../../Pic_dataset/'

path=image_path
path='../../../../../Pic_dataset/'


print("The 4 chanels correspond to the difference between an image and the image just before. In sense add this 4TH chanels could be considered as to add a temporel chanels.")
print("the difference between the 2 pictures is made in grey colors.")
print("If you want to apply the conversion of images to gray and the difference that follows type HSV.")
print("If you want to apply filters before converting to gray, type GBR.")
color_space_diff=input()
print("Type train or test wethever you want to rec picture with a fourth chanel for train or test sample")
print( "I advise you to beguin with small proportion to be sure that your computer has enough memory get pictures")
Sample=input(" Please type train or test " )
test_size = float(input("Type a number between 0.1 and 0.5 to indicate the proportion of data you want put in test sample : "))


Minimum_Number_Class=100

Images=pd.read_csv(Mat_path+"Images.csv")
Images=fn.eliminate_small_categories(Images,Minimum_Number_Class)
images_test=list(Images["filename"].unique())



base_train,base_test= train_test_split(Images,stratify=Images["classe"], test_size=test_size,random_state=42)
if Sample=="train":
    base=base_train
    del Images,base_test
elif Sample=="test":
    base=base_test
    del Images,base_train  
gc.collect()




start=time.time()
X=fn.get_X(base[:2],path,color_space_diff)
Y=fn.get_Y(base[:2])
end=time.time()
print(end-start)
save('imagettes4C_'+color_space_diff+'_'+Sample +'.npy', X)
#save('Y'+color_space_diff+'_'+Sample +'.npy', Y)
gc.collect()

