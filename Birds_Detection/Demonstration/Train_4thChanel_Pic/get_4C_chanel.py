#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:30:46 2020

@author: marcpozzo
"""




#Import libraries
import pandas as pd
import functions_4C as fn
import time
from sklearn.model_selection import train_test_split
import gc
from numpy import save

Mat_path="../../Materiels/"

path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/"#Where you pictures are
liste_folders=['./DonneesPI/timeLapsePhotos_Pi1_0','./DonneesPI/timeLapsePhotos_Pi1_1','./DonneesPI/timeLapsePhotos_Pi1_2','./DonneesPI/timeLapsePhotos_Pi1_3','./DonneesPI/timeLapsePhotos_Pi1_4']


print("The 4 chanels correspond to the difference between an image and the image just before. In sense add this 4TH chanels could be considered as to add a temporel chanels.")
print("the difference between the 2 pictures is made in grey colors.")
print("If you want to apply the conversion of images to gray and the difference that follows type HSV.")
print("If you want to apply filters before converting to gray, type GBR.")
type_of_diff=input()
print("Type train or test wethever you want to rec picture with a fourth chanel for train or test sample")
print( "I advise you to beguin with small proportion to be sure that your computer has enough memory get pictures")
Sample=input(" Please type train or test " )
test_size = input("Type a number between 0.1 and 0.5 to indicate the proportion of data you want put in test sample : ")





imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=fn.open_table(imagettes,liste_folders)
liste_name_test,filename_liste=list(fn.get_liste_name_test(imagettes))
imagettes["filename"]=list(fn.get_liste_name_test(imagettes))[0]


base_train,base_test= train_test_split(imagettes,stratify=imagettes["classe"], test_size=test_size,random_state=42)
if Sample=="train":
    base=base_train
    del imagettes,base_test
elif Sample=="test":
    base=base_test
    del imagettes,base_train  
gc.collect()




start=time.time()
X,Y_train,base_train_trans=fn.get_X_Y(base,diff=type_of_diff)
end=time.time()
print(end-start)
save('imagettes4C_'+type_of_diff+'_'+Sample +'.npy', X)
gc.collect()
