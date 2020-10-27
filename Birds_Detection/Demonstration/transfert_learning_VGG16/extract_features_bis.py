#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:43:09 2020

@author: marcpozzo
"""


#This scripts extract the output of VGG16 
#In the next script train.py a training is making with these outputs
from keras.applications import VGG16
import functions_VGG as fn
import pandas as pd




#Paramètres par défaut
data_path="../../../.."
Mat_path="../../Materiels/"
path_folder=data_path+"DonneesPI/"
fichierClasses= Mat_path+"Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False) # table of species into classes 




# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)


#Select only animals categories

imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=fn.to_reference_labels (imagettes,"classe",frame)
folders_to_keep=['./DonneesPI/timeLapsePhotos_Pi1_4','./DonneesPI/timeLapsePhotos_Pi1_3','./DonneesPI/timeLapsePhotos_Pi1_2','./DonneesPI/timeLapsePhotos_Pi1_1','./DonneesPI/timeLapsePhotos_Pi1_0']
imagettes=imagettes[imagettes["path"].isin(folders_to_keep)]
liste_animals=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes_animals=imagettes[imagettes["classe"].isin(liste_animals)]




#Get train and test sets
train,test=fn.get_train_test_sets(imagettes, imagettes_animals, model, data_path)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)