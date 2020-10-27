#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:43:09 2020

@author: marcpozzo
"""


#This scripts extract the output of VGG16 
#In the next script train.py a training is making with these outputs
from keras.applications import VGG16
import functions_VGG16 as fn
import pandas as pd
from sklearn.model_selection import train_test_split



#Paramètres par défaut
data_path="../../../.."
data_path='../../../../Pic_dataset/'
mat_path="../../Materiels/"




# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)



Images=pd.read_csv(mat_path+"images.csv")
imagettes_=list(Images["filename"].unique())
imagettes_train_,imagettes_test_=train_test_split(imagettes_,test_size=0.2,random_state=42)
Imagettes_train=Images[Images["filename"].isin(imagettes_train_)]
Imagettes_test=Images[Images["filename"].isin(imagettes_test_)]

Features_train=fn.get_tables(Imagettes_train,model,imagettes_train_,data_path)
Features_test=fn.get_tables(Imagettes_test,model,imagettes_test_,data_path)


#Get train and test sets
Features_train.to_csv('train.csv', index=False)
Features_test.to_csv('test.csv', index=False)