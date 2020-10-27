#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:43:09 2020

@author: marcpozzo
"""


#This scripts extract the output of VGG16 
#In the next script train.py a training is making with these outputs
from keras.applications import VGG16
import functions_VGG_bis as fn
import pandas as pd
from sklearn.model_selection import train_test_split



#Paramètres par défaut
data_path="../../../.."
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"




# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)



images=pd.read_csv(Mat_path+"images.csv")
imagettes=images
liste_imagettes=list(images["filename"].unique())
liste_imagettes_train,liste_imagettes_test=train_test_split(liste_imagettes,test_size=0.2,random_state=42)
imagettes_train=images[images["filename"].isin(liste_imagettes_train)]
imagettes_test=images[images["filename"].isin(liste_imagettes_test)]

tableau_features_train=fn.get_tables(imagettes_train,model,liste_imagettes_train[:2],data_path)



#Get train and test sets
#train,test=fn.get_train_test_sets(imagettes, imagettes_animals, model, data_path)
#train.to_csv('train.csv', index=False)
#test.to_csv('test.csv', index=False)