#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""


from os import chdir
chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/bin")
import pandas as pd
import functions as fn
import os
from os.path import basename, join
import numpy as np
import time
from tensorflow.keras.models import load_model


neurone_feature="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"
CNNmodel  = load_model(neurone_feature,compile=False)

path_folder="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/"


#Paramètres par défaut
path="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises"
fichierClasses= "/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile

#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")

imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 


#imagettes=imagettes[ (imagettes["filename"]!='image_2019-04-30_18-17-14.jpg') and (imagettes["filename"]!='image_2019-04-18_17-56-42.jpg') ]


liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]

liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_1']



Diff_image_FP_by_folder,Diff_image_animals_by_folder,Diff_image_image_total_by_folder,nb_FP_liste,nb_TP_liste,nb_FP_liste_thresh,nb_TP_liste_thresh=[[] for i in range(7)]


start=time.time()

for folder in liste_folders:

    chdir(path+folder)
    liste_image_ref,estimates_FP_liste,estimates_match_brids_liste,Diff_image_FP_liste,Diff_image_animals_liste,Diff_image_image_total_liste=[[] for i in range(6)]
    

    for r, d, f in os.walk(path+folder):
        for file in f:
            if '.jpg' in file:
                liste_image_ref.append(basename(join(r, file)))
                
    liste_image_ref.sort()           
    path_images=folder+"/"   
    folder_choosen="."+folder
    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ] 
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    print("nombre d'images pour le dossier",len(liste_name_test))
    folder_name=folder[-5:]
    (nb_folder_TP,nb_folder_FP,nb_TP_folder_thresh,nb_FP_folder_thresh)=(0,0,0,0)
    for name_test in liste_name_test:
               
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=
        imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=base_4C_poly(name_test,name_ref,
                                                                                                                        folder,CNNmodel,blockSize=25,thresh=0.9,
                                                                                                                        blurFact=17,chanels=3,contrast=-5,maxAnalDL=20,method="light",mask=True,focus="bird_large")
        
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
print("le temps pris en minutes",duree/60)

print("nombre total de TP :", np.sum(nb_TP_liste))
print("nombre total de FP :", np.sum(nb_FP_liste))

print("nombre total de TP seuil 0.55 :", np.sum(nb_TP_liste_thresh))
print("nombre total de FP seuil 0.55:", np.sum(nb_FP_liste_thresh))



"""
Tout dernier programme
nombre total de TP : 213
nombre total de FP : 131
nombre total de TP seuil 0.55 : 213
nombre total de FP seuil 0.55: 125



nombre total de TP : 212
nombre total de FP : 116
nombre total de TP seuil 0.55 : 212
nombre total de FP seuil 0.55: 114

nombre TP par dossier [199, 319, 1, 230, 44]
nombre FP par dossier [27, 2157, 2, 420, 503]
nombre de Tp au dessus du seuil de 0.9 [196, 315, 1, 212, 41]
nombre de Fp au dessus du seuil de 0.9 [26, 2136, 2, 386, 466]
ns 25
nombre TP par dossier [248]
nombre FP par dossier [472]
"""


"""
nombre TP par dossier [104, 343, 19, 207, 93]
nombre FP par dossier [14, 170, 2, 38, 61]
nombre de Tp au dessus du seuil de 0.9 [50, 135, 11, 20, 26]
nombre de Fp au dessus du seuil de 0.9 [8, 58, 0, 0, 11]


large
nombre total de TP : 215
nombre total de FP : 83
nombre total de TP seuil 0.55 : 212
nombre total de FP seuil 0.55: 77


"""
