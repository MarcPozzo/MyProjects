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


    

#dic="latlong":{"lat":"0","long":"0","mapImage":"maps/SIC.png"},"shotFile":"photos/stillCurrent.jpeg?t=1592890232186","sizeShot":[616,820],"map":{"markers":[{"id":"1","coord":{"lat":14,"lng":16}},{"id":"2","coord":{"lat":138,"lng":11}},{"id":"3","coord":{"lat":288,"lng":10}},{"id":"4","coord":{"lat":20,"lng":89}},{"id":"5","coord":{"lat":148,"lng":104}},{"id":"6","coord":{"lat":304,"lng":94}},{"id":"7","coord":{"lat":17,"lng":185}},{"id":"8","coord":{"lat":154,"lng":170}},{"id":"9","coord":{"lat":317,"lng":185}},{"id":"10","coord":{"lat":23,"lng":281}},{"id":"11","coord":{"lat":164,"lng":301}},{"id":"C","coord":{"lat":234,"lng":331}},{"id":"12","coord":{"lat":324,"lng":302}},{"id":"C","coord":{"lat":90,"lng":286}},{"id":"13","coord":{"lat":30,"lng":384}},{"id":"14","coord":{"lat":174,"lng":392}},{"id":"15","coord":{"lat":327,"lng":396}},{"id":"16","coord":{"lat":28,"lng":524}},{"id":"17","coord":{"lat":174,"lng":523}},{"id":"18","coord":{"lat":323,"lng":528}}]

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

blurFact_liste=[17]
for blurFact in blurFact_liste:
    path_save="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/listes/Comp_imgen_imanote/BF"+str(blurFact)+"/"


    #with open(path_save+"image_no_caught_BS"+str(blockSize)+".txt", "rb") as fp:   # Unpickling
    #    No_caught = pickle.load(fp)
        
    #liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1']
    surface_carre_gen_tous_fichiers=[]
    surface_anote_liste_tous_fichiers=[]
    
    
    
    for folder in liste_folders:
        surface_carre_gen_par_fichier=[]
        surface_anote_liste_par_fichier=[]
    
        
        liste_name_test,liste_image_ref=fns.get_liste_name_test(folder,imagettes)
    
        #liste_name_test=list(set(liste_name_test)-set(No_caught)) 
        nombre_animaux=0
        for name_test in liste_name_test:
            #name_test=No_caught[-1]
    
            
            index_of_ref=liste_image_ref.index(name_test)-1
            name_ref=liste_image_ref[index_of_ref]
            print(name_test,name_ref)
            
    
            surface_carre_gen_liste, surface_anote_liste=fns.extract_size_imagettes(name_test,name_ref,folder,blockSize=17,blurFact=17)
           
            
            surface_carre_gen_par_fichier+=surface_carre_gen_liste
            surface_anote_liste_par_fichier+=surface_anote_liste
            
            
    
        
        
        
        surface_carre_gen_tous_fichiers.append(surface_carre_gen_par_fichier)
        surface_anote_liste_tous_fichiers.append(surface_anote_liste_par_fichier)
    
        
        
    
    with open(path_save+"surface_carre_gen_tous_fichiers_BF"+str(blurFact)+".txt", "wb") as fp:   #Pickling
        pickle.dump(surface_carre_gen_tous_fichiers, fp)
    
     
    with open(path_save+"surface_anote_liste_tous_fichiers_BF"+str(blurFact)+".txt", "wb") as fp:   #Pickling
        pickle.dump(surface_anote_liste_tous_fichiers, fp)
    
"""
sudo docker-compose up
positionning.php
vim bashServer.py 
setup.md
"""


