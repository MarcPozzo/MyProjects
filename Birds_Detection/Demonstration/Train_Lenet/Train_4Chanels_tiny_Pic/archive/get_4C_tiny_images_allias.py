#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:55:46 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:43:44 2020

@author: marcpozzo
"""

#This script was moved keep attention to change the path ....
#This script create a new chanels from the difference of one tiny picture and tiny picture taken juste before.

#Import libraries
import pandas as pd
import functions_4C as fn
from sklearn.model_selection import train_test_split
import gc
import os
from numpy import save
Mat_path="../../../Materiels/"

#Load Table
#image_path='../../../../../Pic_dataset/'
tiny_image_path='../../../../../Tiny_images/'
image_path_to_save='../../../../../4C_tiny_picture/'
Images=pd.read_csv(Mat_path+"tiny_Images.csv")



print("The 4 chanels correspond to the difference between an image and the image just before. In sense add this 4TH chanels could be considered as to add a temporel chanels.")
print("the difference between the 2 pictures is made in grey colors.")
print("If you want to apply the conversion of images to gray and the difference that follows type HSV.")
print("If you want to apply filters before converting to gray, type BGR.")
color_space_diff=input("Please type HSV or BGR ")
image_path_to_save=image_path_to_save+color_space_diff+"/"
os.mkdir(image_path_to_save)
print("Type train or test wethever you want to rec picture with a fourth chanel for train or test sample")
print( "I advise you to beguin with small proportion to be sure that your computer has enough memory get pictures")

test_size = float(input("Type a number between 0.1 and 0.5 to indicate the proportion of data you want put in test sample : "))









Sample=input(" Please type train or test " )
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





base_train,base_test= train_test_split(Images,stratify=Images["classe"], test_size=test_size,random_state=42)

X=fn.get_X(base,tiny_image_path,color_space_diff)
Y=fn.get_Y(base)





for i in range(len(X)):
    name=base["imagetteName"].iloc[i][:-4]
    save(image_path_to_save+name+color_space_diff+'_'+Sample +'.npy', X[i])

save(image_path_to_save+'labels4C_'+color_space_diff+'_'+Sample +'.npy', Y)
gc.collect()
