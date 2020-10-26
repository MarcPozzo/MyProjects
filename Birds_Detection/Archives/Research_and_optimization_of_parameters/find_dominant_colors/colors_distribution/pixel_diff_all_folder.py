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






liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]


Diff_image_FP_by_folder=[]
Diff_image_animals_by_folder=[]
Diff_image_image_total_by_folder=[]


liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_1']
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

    for name_test in liste_name_test:
        

        
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        


        Diff_image_total=fn.extract_dist_by_type_essai(name_test,name_ref,folder,CNNmodel,6,mask=False)

        #Diff_image_FP_liste.append(Diff_image_FP)
        #Diff_image_animals_liste.append(Diff_image_animals)
        Diff_image_image_total_liste.append(Diff_image_total)

    #Diff_image_FP_by_folder.append(Diff_image_FP_liste)
    #Diff_image_animals_by_folder.append(Diff_image_animals_liste)
    #Diff_image_image_total_by_folder.append(Diff_image_image_total_liste)
    



with open(save_path+"no_an_fold1.txt", "wb") as fp:   #Pickling
    pickle.dump(Diff_image_image_total_liste, fp)



import pickle

save_path="/home/marcpozzo/Desktop/c3po_interface_mark/Materiels/listes/Pixel_hsv/less_fp_200/"

with open(save_path+"lfp200_fp.txt", "wb") as fp:   #Pickling
    pickle.dump(Diff_image_FP_by_folder, fp)

with open(save_path+"lfp200_an.txt", "wb") as fp:   #Pickling
    pickle.dump(Diff_image_animals_by_folder, fp)
    


#Load the data

with open(save_path+"lfp200_fp.txt", "rb") as fp:   # Unpickling
    Diff_image_FP_by_folder = pickle.load(fp)
 
with open(save_path+"lfp200_an.txt", "rb") as fp:   # Unpickling
    Diff_image_animals_by_folder = pickle.load(fp)

a="/home/marcpozzo/Desktop/c3po_interface_mark/Materiels/listes/Pixel_hsv/less_fp_200/lfp200_l_imagettes.txt"
with open(a, "rb") as fp:   # Unpickling
    Diff_image_image_total_by_folder = pickle.load(fp)


flat_list_total = [item for sublist in Diff_image_image_total_by_folder[0]  for item in sublist]



flat_an = [item for sublist in Diff_image_animals_by_folder for item in sublist]
flat_fp = [item for sublist in Diff_image_FP_by_folder  for item in sublist]

flat_an = [item for sublist in flat_an for item in sublist]
flat_fp = [item for sublist in flat_fp  for item in sublist]


flat_an = [item for sublist in flat_an for item in sublist]
flat_fp = [item for sublist in flat_fp  for item in sublist]


print("liste animals")
print("moyenne",np.mean(flat_an))
print("medfiane",np.median(flat_an))
print("standart erreur",np.std(flat_an))

print("liste faux positifs")
print("moyenne",np.mean(flat_fp))
print("medfiane",np.median(flat_fp))
print("standart erreur",np.std(flat_fp))



#Maintenant dossier par dossier



print("folder1")
flat_list_an = [item for sublist in Diff_image_animals_by_folder[2] for item in sublist]
flat_list_fp = [item for sublist in Diff_image_FP_by_folder[2]  for item in sublist]
flat_list_fp = [item for sublist in flat_list_fp  for item in sublist]
flat_list_an = [item for sublist in flat_list_an for item in sublist]

#plt.hist(flat_list_an );
#plt.hist(flat_list_fp );
#plt.hist(flat_list_total)

print("liste animals")
print("moyenne",np.mean(flat_list_an))
print("medfiane",np.median(flat_list_an))
print("standart erreur",np.std(flat_list_an))

print("liste faux positifs")
print("moyenne",np.mean(flat_list_fp))
print("medfiane",np.median(flat_list_fp))
print("standart erreur",np.std(flat_list_fp))

print("liste total")
print("moyenne",np.mean(flat_list_total))
print("medfiane",np.median(flat_list_total))

np.mean((Diff_image_animals_by_folder[0]))

#end=time.time()
#end-start



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
    
    
    
    
extract_distance(path_images,name_test,name_ref,folder)
extract_dist_by_type_shorter(name_test,name_ref,folder,CNNmodel,numb_classes)