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
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0']



Diff_image_FP_by_folder,Diff_image_animals_by_folder,Diff_image_image_total_by_folder,nb_FP_liste,nb_TP_liste,nb_FN1_liste,nb_FN2_liste=[[] for i in range(7)]

FP_by_thresh=[]
FN1_by_thresh=[]
FN2_by_thresh=[]
nb_folder_diff_thresh=[]
nb_folder_oiseau_thresh=[]
#tresh_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,0.99]
tresh_list=[0.2,0.95]
tresh_list=[0,0.4,0.6,0.7,0.9]
#tresh_list=[0]
#tresh_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.92,0.95,0.97,0.999]
#tresh_list=[0]
#tresh_list=[0.3,0.6,0.9]
start=time.time()

for thresh_t in tresh_list:
    nb_FP_liste,nb_TP_liste,nb_FN1_liste,nb_FN2_liste=[[] for i in range(4)]
    nb_folder_diff_liste=[]
    nb_folder_oiseau_liste=[]
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
        (nb_folder_TP,nb_folder_FP,nb_folder_FN1,nb_folder_FN2,nb_folder_diff,nb_folder_oiseau)=(0,0,0,0,0,0)

        for name_test in liste_name_test:
                   
            index_of_ref=liste_image_ref.index(name_test)-1
            name_ref=liste_image_ref[index_of_ref]
            print(name_test,name_ref)
            
            imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.base_4C_poly(name_test,name_ref,
                                                                                                                            folder,CNNmodel,blockSize=25,thresh=0,
                                                                                                                            blurFact=17,chanels=3,contrast=-8,maxAnalDL=thresh_t,method="light",mask=True,focus="bird_prob")
    
             
            
            nb_TP_birds=len(TP_birds)
            nb_FN1=nb_oiseaux-nb_TP_birds
            nb_FN2=len(liste_Diff_birds)-nb_TP_birds
            
            nb_FP=len(FP)
    
     
            nb_folder_FN1+= nb_FN1
            nb_folder_FN2+= nb_FN2
            nb_folder_FP+=nb_FP
            nb_folder_diff+=len(liste_Diff_birds)
            nb_folder_oiseau+=nb_oiseaux
    

        nb_FP_liste.append(nb_folder_FP)
        nb_FN1_liste.append(nb_folder_FN1)
        nb_FN2_liste.append(nb_folder_FN2)
        nb_folder_diff_liste.append(nb_folder_diff)
        nb_folder_oiseau_liste.append(nb_folder_oiseau)
        
    FP_thresh=np.sum(nb_FP_liste)
    FN1_thresh=np.sum(nb_FN1_liste)
    FN2_thresh=np.sum(nb_FN2_liste)

    FP_by_thresh.append(FP_thresh)
    FN1_by_thresh.append(FN1_thresh)
    FN2_by_thresh.append(FN2_thresh)

end=time.time()
duree=end-start 
#print("nombre TP par dossier",nb_TP_liste)  
#print("nombre FP par dossier",FP_by_thresh) 

print("le temps pris en minutes",duree/60)

print("nombre total de FN1 :", FN1_by_thresh)
print("nombre total de FN2 :", FN2_by_thresh)
print("nombre total de FP :", FP_by_thresh)




"""
nombre total de FN1 : [513]
nombre total de FN2 : [2]
nombre total de FP : [4518]
"""



"""
Maintenant pour ssim

[0,0.4,0.5,0.6,0.7,0.8,0.9]
nombre total de FN1 : [1397, 1397,1399, 1678, 2137, 2420]

nombre total de FP : [6362, 6362,6335, 5485,  870,    0]


maxAnalDL=0.6
nombre total de FN1 : [1678, 1774, 1806, 1833, 1853, 1870, 1881, 1915, 1952, 1990]
nombre total de FN2 : [2, 98, 130, 157, 177, 194, 205, 239, 276, 314]
nombre total de FP : [5485, 936, 733, 647, 572, 536, 494, 452, 388, 317]



[]





"""






"""


tresh_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]




thresh=0
maxAnld==           [0,0.1,0.2,0.3,0.4,0.5,0.6,             0.7,0.72,0.74,                 0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,0.99]
nombre total de FN1 : [918, 918, 918, 918, 918, 921, 1292,    1990, 2138, 2274,               2395, 2411, 2420, 2420, 2420, 2420, 2420, 2420, 2420, 2420, 2420, 2420, 2420]
nombre total de FN2 : [4, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nombre total de FP : [431875, 431875, 431875, 431875, 431875, 423689,      302820,69157, 38772,        19121, 8184, 3016, 426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]








nombre total de FN1 : [311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 407]
nombre total de FN2 : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nombre total de FP : [1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 139]





nombre TP par dossier [104, 343, 19, 207, 93]
nombre FP par dossier [14, 170, 2, 38, 61]
nombre de Tp au dessus du seuil de 0.9 [50, 135, 11, 20, 26]
nombre de Fp au dessus du seuil de 0.9 [8, 58, 0, 0, 11]
"""