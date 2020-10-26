#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""
#numpy 1.18.1

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/bin")
import pandas as pd
#import functions_court as fn
import functions as fns

import os
from os.path import basename, join
import numpy as np
import ast
#from keras.models import Model, load_model
import cv2
from scipy import stats 
import tensorflow  
import tensorflow.keras.models      
from tensorflow.keras.models import load_model
import pickle
import time
from tensorflow import keras
from keras.applications import VGG16
 

#Paramètres à choisir

#neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"
#neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/match+fp_db/lessfp_100ep"
#CNNmodel  = load_model(neurone_feature,compile=False)
#CNNmodel  = load_model(neurone_feature)

# load the trained model
c3poFolder="/home/marcpozzo/Desktop/c3po_interface_mark/Materiels/"
#output/RL_annotation_model"
#Model1 = joblib.load(c3poFolder+"output/model.cpickle")
#filtre_RL = joblib.load(c3poFolder+"output/RL_annotation_model")


#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes=fns.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']


#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0' ]
model = VGG16(weights="imagenet", include_top=False)
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_' +str(i) for i in range(5)]




liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4' ]
#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_1']
nb_FP_liste=[]
nb_TP_liste=[]
nb_FP_liste_thresh=[]
nb_TP_liste_thresh=[]


class_num=2
start=time.time()
for folder in liste_folders:
    liste_name_test,liste_image_ref=fns.get_liste_name_test(folder,imagettes)

    Diff_image_FP_liste=[]
    Diff_image_animals_liste=[]
    Diff_image_image_total_liste=[]
    
    nb_folder_TP=0
    nb_folder_FP=0
    nb_TP_folder_thresh=0
    nb_FP_folder_thresh=0
    for name_test in liste_name_test:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        

       
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=base_Khalid(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="No_filtre",blockSize=17,blurFact=17,chanels=3,contrast=-8,thresh=0.5)
        imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=fns.base_Khalid_bis(name_test,name_ref,folder,filtre_choice="No_filtre",thresh=0.5)
        
        nb_TP_birds=len(TP_birds)
        nb_FP=len(FP)
        nb_TP_thresh=len(TP_estimates)
        nb_FP_thresh=len(FP_estimates)
 
        nb_folder_TP+= nb_TP_birds
        nb_folder_FP+=nb_FP
        nb_TP_folder_thresh+=nb_TP_thresh
        nb_FP_folder_thresh+=nb_FP_thresh
    
    nb_FP_liste.append(nb_folder_FP)
    nb_TP_liste.append(nb_folder_TP)
    nb_FP_liste_thresh.append(nb_FP_folder_thresh)
    nb_TP_liste_thresh.append(nb_TP_folder_thresh)    
end=time.time()  
duree=end-start

print("nombre TP par dossier",nb_TP_liste)  
print("nombre FP par dossier",nb_FP_liste) 
print("nombre de Tp au dessus du seuil de 0.9",nb_TP_liste_thresh)   
print("nombre de Fp au dessus du seuil de 0.9",nb_FP_liste_thresh)
print("le nombre de minutes que prend le programme est :",duree/60)
        

        
#Resultat #'image_2019-06-15_17-40-01.jpg' pour ce name_test c'est compliqué, mais il faut reconnaitre que oiseau difficie et masque ?
        

#Maintenant essayon avec celle là : image_2019-04-30_18-55-03.jpg