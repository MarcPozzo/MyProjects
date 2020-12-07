#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:43:44 2020

@author: marcpozzo
"""

#This script create a new chanels from the difference of one tiny picture and tiny picture taken juste before.

#Import libraries
import pandas as pd
import functions_4C as fn
from sklearn.model_selection import train_test_split
from numpy import save




#Map data
Mat_path="../../../Materiels/"
tiny_image_path='../../../../../Tiny_images/'
image_path_to_save='../../../../../4C_tiny_picture/'
Images=pd.read_csv(Mat_path+"tiny_Images.csv")


# Ask Parameters
print("The 4 chanels correspond to the difference between an image and the image just before. In sense add this 4TH chanels could be considered as to add a temporel chanels.")
print("the difference between the 2 pictures is made in grey colors.")
print("If you want to apply the conversion of images to gray and the difference that follows type HSV.")
print("If you want to apply filters before converting to gray, type BGR.")
color_space_diff=input("Please type HSV or BGR ")

print("Type train or test wethever you want to rec picture with a fourth chanel for train or test sample")
print( "I advise you to beguin with small proportion to be sure that your computer has enough memory get pictures")

test_size = float(input("Type a number between 0.1 and 0.5 to indicate the proportion of data you want put in test sample : "))



#Get labels and 4th chanels pictures and save them 
base_train,base_test= train_test_split(Images,stratify=Images["classe"], test_size=test_size,random_state=42)

X_train=fn.get_4C_Pic(base_train,tiny_image_path,color_space_diff)
Y_train=fn.get_Y(base_train)
save(image_path_to_save+color_space_diff+'timages4C_train.npy', X_train)
save(image_path_to_save+color_space_diff+'labels_train.npy', Y_train)

X_test=fn.get_4C_Pic(base_test,tiny_image_path,color_space_diff)
Y_test=fn.get_Y(base_test)
save(image_path_to_save+color_space_diff+'timages4C_test.npy', X_test)
save(image_path_to_save+color_space_diff+'labels_test.npy', Y_test)
print("data were saved in",image_path_to_save)



