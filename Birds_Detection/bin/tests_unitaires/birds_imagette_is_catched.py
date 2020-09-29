#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:28 2020

@author: pi




"""
#Ce script propose de vérifier si les carrés d'annotations sont bien repérées par la différence puis après le ou les filtres sous forme de fonction avec un script sourçale et sunthétique

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import functions as fn
import cv2
import pandas as pd
import joblib


output_Images_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/"

#Paramètres à choisir

#Pour filtre quantile



filtre_choice="No_filtre" #"quantile_filtre"#"No_filtre" "RL_filtre"



#Autres Paramètres
path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes_zoom/"

c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface/"
#Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")
filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")



coef_filtre=pd.read_csv("testingInputs/coefs_filtre_RQ.csv")

name2 = "EK000228.JPG"
imageA = cv2.imread("testingInputs/EK000227.JPG")
imageB = cv2.imread("testingInputs/"+name2)
 


#Neural_models=["zoom_0.9:1.3_flip","6c_rob","zoom_1.3","drop_out.50","z1.3"]
name_model="z1.3"
neurone_features=path+name_model
path_anotation="testingInputs/oiseau_lab_Alex.csv"



print(fn.birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2)==True)