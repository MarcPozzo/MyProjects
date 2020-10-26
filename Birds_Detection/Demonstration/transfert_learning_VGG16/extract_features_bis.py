#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:48:29 2020

@author: marcpozzo
"""

#Rajouter preprocess input
#Transfo en dataset
#random table
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
liste_name_test=list(imagettes_animals["filename"].unique())
imagettes=imagettes[imagettes["filename"].isin(liste_name_test)]



tableau_birds_features=fn.get_features_to_df(imagettes,model,liste_name_test,data_path,bird=True)
tableau_other_features=fn.get_features_to_df(imagettes,model,liste_name_test,data_path,bird=False)
tableaux=[tableau_birds_features,tableau_other_features]
tableau_features=pd.concat(tableaux)
tableau_features=tableau_features.sample(frac=1).reset_index(drop=True)
tableau_features.to_csv('X_train.csv', index=False)