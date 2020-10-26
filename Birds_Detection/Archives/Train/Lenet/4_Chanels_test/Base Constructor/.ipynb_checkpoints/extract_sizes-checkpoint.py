#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""


from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/bin")
import pandas as pd
#import functions_court as fn
import functions_netoyees as fns

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

"""
nombre TP par dossier [201]
nombre FP par dossier [911]
nombre de Fp au dessus du seuil de 0.9 [595]
nombre de Tp au dessus du seuil de 0.9 [195]
"""
 

#Paramètres à choisir

#CNNmodel = tensorflow.keras.models.load_model(neurone_feature)

#neurone='/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/148ep_0.5dpt_GBR__6CL_4CH'
neurone='/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/birds_fp_matched/400ep_0.1dpt_HSV_6CL_4CH_smal_fp_birds_matched'
#neurone='/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/200ep_GBR_6CL_4CH'
neurone="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/all_fp+all_birds/10ep_0.1dpt_HSV__6CL_4CH_half_fp_birds_matched"
naurone="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/all_fp+all_birds/40ep_0.3dpt_HSV_6CL_4CH_small_fp_birds_matched"
CNNmodel = keras.models.load_model(neurone)



path_folder="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/"


#Paramètres par défaut
path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
#frame=pd.read_csv(fichierClasses,index_col=False)





#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
#imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=fns.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']


#fns.extract_fp_4C(path_images,name_test,name_ref,folder,CNNmodel,path_images)
#path_images="z"
#fns.extract_fp_4C(name_test,name_ref,folder,CNNmodel)

liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]
#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_4'  ]

Diff_image_FP_by_folder=[]
Diff_image_animals_by_folder=[]
Diff_image_image_total_by_folder=[]


#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_1']

TP_size_liste=[]
FP_size_liste=[]
birds_size_liste=[]

for folder in liste_folders:

    chdir(path+folder)
    liste_image_ref = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path+folder):
        for file in f:
            if '.jpg' in file:
                liste_image_ref.append(basename(join(r, file)))
                
                
    path_images=folder+"/"
    #folder="/DonneesPI/timeLapsePhotos_Pi1_1"
    folder_choosen="."+folder
    #imagettes_PI_0=imagettes[imagettes["path"]=="/DonneesPI/timeLapsePhotos_Pi1_2" ]

    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ]
    
    #Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer


    
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    estimates_FP_liste=[]
    estimates_match_brids_liste=[]
    folder_name=folder[-5:]
    
    
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
        

        TP_size,birds_size,FP_size=extract_size(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="quantile_filtre")
        TP_size_liste+=TP_size
        birds_size_liste+=birds_size
        FP_size_liste+=FP_size

print("la taille moyenne des oiseaux prédits est",)

import seaborn as sns 

sns.distplot(np.log(birds_size_liste)/np.log(10), hist=True, kde=True, 
             bins=5, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

   
 
"""
sudo docker-compose up
positionning.php
vim bashServer.py 
setup.md
"""