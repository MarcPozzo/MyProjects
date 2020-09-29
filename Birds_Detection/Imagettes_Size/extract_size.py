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

path_save="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/"
#/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/

with open(path_save+"image_no_caught_BS35.txt", "rb") as fp:   # Unpickling
    No_caught = pickle.load(fp)
    
neurone="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/all_fp+all_birds/10ep_0.1dpt_HSV__6CL_4CH_half_fp_birds_matched"
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



liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]

#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1']
surface_carre_gen_tous_fichiers=[]
surface_anote_liste_tous_fichiers=[]


for folder in liste_folders:
    surface_carre_gen_par_fichier=[]
    surface_anote_liste_par_fichier=[]
    surface_carre_gen_ce_fichier=[]
    surface_anote_liste_ce_fichier=[]
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
    folder_name=folder[-5:]
    liste_name_test=list(set(liste_name_test)-set(No_caught)) 
    nombre_animaux=0
    for name_test in liste_name_test:
        

        
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        

        surface_carre_gen_liste, surface_anote_liste=fns.extract_size_imagettes(name_test,name_ref,folder,CNNmodel)
        
        surface_carre_gen_par_fichier+=surface_carre_gen_liste
        surface_anote_liste_par_fichier+=surface_anote_liste
        
        

    
    
    
    surface_carre_gen_tous_fichiers.append(surface_carre_gen_par_fichier)
    surface_anote_liste_tous_fichiers.append(surface_anote_liste_par_fichier)
    #print("surface_carre_gen_tous_fichiers :",surface_carre_gen_tous_fichiers)
    #print("surface_anote_liste_tous_fichiers :", surface_anote_liste_tous_fichiers)
    
path_save="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/BS35/"

with open(path_save+"surface_carre_gen_tous_fichiers_BS35.txt", "wb") as fp:   #Pickling
    pickle.dump(surface_carre_gen_tous_fichiers, fp)

 
with open(path_save+"surface_anote_liste_tous_fichiers_BS35.txt", "wb") as fp:   #Pickling
    pickle.dump(surface_anote_liste_tous_fichiers, fp)
    
"""
sudo docker-compose up
positionning.php
vim bashServer.py 
setup.md
"""


