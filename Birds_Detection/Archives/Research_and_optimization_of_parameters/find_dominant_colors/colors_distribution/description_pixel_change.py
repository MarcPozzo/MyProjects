#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""


from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/bin")
import pandas as pd
import functions as fn

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



#Paramètres à choisir

path_neurone="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/match+fp_db/"
neurone="lessfp_200ep"
neurone_feature=path_neurone+neurone
CNNmodel = tensorflow.keras.models.load_model(neurone_feature)



path_folder="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/"


#Paramètres par défaut
path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
#frame=pd.read_csv(fichierClasses,index_col=False)





#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']


folder='timeLapsePhotos_Pi1_4'

folder="/DonneesPI/"+folder
chdir(path+folder)
liste_image_ref = []
for r, d, f in os.walk(path+folder):
    for file in f:
        if '.jpg' in file:
            liste_image_ref.append(basename(join(r, file)))
                             
path_images=folder+"/"
folder_choosen="."+folder
imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    
    

    
liste_name_test=list(imagettes_folder["filename"].unique())

(FP_birds_thresh_total,birds_pr_birds_thresh_total)=(0,0)
name_test = "image_2019-06-14_15-46-54.jpg"
name_test = "image_2019-06-14_15-47-11.jpg"
index_of_ref=liste_image_ref.index(name_test)-1
name_ref=liste_image_ref[index_of_ref]
print(name_test,name_ref)

        
#colors_imagettes(name_test,name_ref,folder,CNNmodel,numb_classes=6,thresh=0.9,filtre_choice="No_filtre",coverage_threshold=0.01,thresh_active=True,to_Select="FP")



start=time.time()

Diff_image_FP,Diff_image_animals,Diff_image_total=extract_dist_by_type_essai(name_test,name_ref,folder,CNNmodel,6,mask=False)
end=time.time()
end-start





liste_diff_imagettes=extract_distance(path_images,name_test,name_ref,folder)



def encadrement_liste(bm_3,interval):
    bm_3_filtre=[i for i in bm_3 if (i>-interval and i<interval)]
    return bm_3_filtre


diff_mm=[a_i - b_i for a_i, b_i in zip(moyenne_tot, median_tot)]
diff_mm2=[a_i - b_i for a_i, b_i in zip(moyenne_tot, median_tot)]


flat_list_an = [item for sublist in Diff_image_animals for item in sublist]
flat_list_fp = [item for sublist in Diff_image_FP for item in sublist]
flat_list_total = [item for sublist in Diff_image_total for item in sublist]

plt.hist(flat_list_an )
plt.hist(flat_list_fp )

#len(flat_list_total)
#35 000 compris entre 0 et 0.5 sur 101 000
plt.hist(flat_list_total )


plt.hist(encadrement_liste(flat_list_an,200))


moyenne_tot=[]
median_tot=[]

for i in range(len(Diff_image_total)):
    moyenne_tot.append(np.mean(Diff_image_total[i]))
    median_tot.append(np.median(Diff_image_total[i]))
    