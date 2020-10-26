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


neurone='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes/200_ep_sans_transfo_bis'
CNNmodel = keras.models.load_model(neurone)




#Paramètres par défaut

#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
#imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=fns.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']



liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]


#blockSize_liste=[12,17,23,27,37,41]

blockSize_liste=[17,53]
for blockSize in blockSize_liste:
    path_save="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/"+"BS"+str(blockSize)+"/"


    #with open(path_save+"image_no_caught_BS"+str(blockSize)+".txt", "rb") as fp:   # Unpickling
    #    No_caught = pickle.load(fp)
        
    #liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1']
    surface_carre_gen_tous_fichiers=[]
    surface_anote_liste_tous_fichiers=[]
    surface_couverte_liste_tous_fichiers=[]
    
    
    for folder in liste_folders:
        surface_carre_gen_par_fichier=[]
        surface_anote_liste_par_fichier=[]
        surface_couverte_liste_par_fichier=[]
        
        liste_name_test,liste_image_ref=fns.get_liste_name_test(folder,imagettes)
    
        #liste_name_test=list(set(liste_name_test)-set(No_caught)) 
        nombre_animaux=0
        for name_test in liste_name_test:
            #name_test=No_caught[-1]
    
            
            index_of_ref=liste_image_ref.index(name_test)-1
            name_ref=liste_image_ref[index_of_ref]
            print(name_test,name_ref)
            
    
            surface_carre_gen_liste, surface_anote_liste,imagette_coverage_liste=extract_size_imagettes(name_test,name_ref,folder,blockSize=blockSize,blurFact=17)
           
            
            surface_carre_gen_par_fichier+=surface_carre_gen_liste
            surface_anote_liste_par_fichier+=surface_anote_liste
            surface_couverte_liste_par_fichier+=imagette_coverage_liste
            
    
        
        
        
        surface_carre_gen_tous_fichiers.append(surface_carre_gen_par_fichier)
        surface_anote_liste_tous_fichiers.append(surface_anote_liste_par_fichier)
        surface_couverte_liste_tous_fichiers.append(surface_couverte_liste_par_fichier)
        
        
    path_save="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/BS"+str(blockSize)+"/"
    
    
    with open(path_save+"surface_intersection_BS"+str(blockSize)+"_BF17.txt", "wb") as fp:   #Pickling
        pickle.dump(surface_carre_gen_tous_fichiers, fp)
        
    with open(path_save+"surface_carre_gen_tous_fichiers_BS"+str(blockSize)+"_BF17.txt", "wb") as fp:   #Pickling
        pickle.dump(surface_carre_gen_tous_fichiers, fp)
    
     
    with open(path_save+"surface_anote_liste_tous_fichiers_BS"+str(blockSize)+"_BF17.txt", "wb") as fp:   #Pickling
        pickle.dump(surface_anote_liste_tous_fichiers, fp)
    
"""
sudo docker-compose up
positionning.php
vim bashServer.py 
setup.md
"""

imagettes["classe"][imagettes["filename"]=="image_2019-04-30_18-28-05.jpg"]
 