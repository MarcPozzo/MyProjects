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


#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
#imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=fns.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']



liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]


blockSize_liste=[12,17,23,27,37,41]
for blockSize in blockSize_liste:
    No_caught_par_fichier=[]
    nb_oiseau_par_fichier=[]
    No_caught=[]
    
    for folder in liste_folders:
        No_caught_fichier=[]
        liste_name_test,liste_image_ref=fns.get_liste_name_test(folder,imagettes)
        
        
        nombre_animaux=0
        for name_test in liste_name_test:
            #print(name_test)
            
    
            #name_test='image_2019-05-24_17-36-38.jpg'
            index_of_ref=liste_image_ref.index(name_test)-1
            name_ref=liste_image_ref[index_of_ref]
            print(name_test,name_ref)
            
    
            liste_Diff_animals,no_caught=fns.num_an_caught(name_test,name_ref,folder,blockSize=31)
            
            
            nombre_animaux+=len(liste_Diff_animals)
            No_caught+=no_caught
            No_caught_fichier+=no_caught
        
    
        No_caught_par_fichier.append(len(No_caught_fichier))
        nb_oiseau_par_fichier.append(nombre_animaux)
        print("///////////////////////////////////////")
        print("nb_oiseau_par_fichier",nb_oiseau_par_fichier)
        print("nb_imagettes sans aucun animaux idenifiés",No_caught_par_fichier)
        print("///////////////////////////////////////")
        
    print("nombre total d'oiseaux:",np.sum(nb_oiseau_par_fichier))
    
    path_save="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/BS"+str(blockSize)+"/"
    
    with open(path_save+"image_no_caught_BS"+str(blockSize)+".txt", "wb") as fp:   #Pickling
        pickle.dump(No_caught, fp)


No_caught[0]


"""
Pour 25
nb_oiseau_par_fichier [258, 835, 78, 479, 391]
nb_imagettes sans aucun animaux idenifiés [9, 55, 18, 75, 42]
///////////////////////////////////////
nombre total d'oiseaux: 2041
"""


"""
sudo docker-compose up
positionning.php
vim bashServer.py 
setup.md
"""