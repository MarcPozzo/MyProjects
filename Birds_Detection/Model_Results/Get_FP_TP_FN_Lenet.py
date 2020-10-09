#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""



import pandas as pd
import functions_Lenet_VGG as fn
import os
from os.path import basename, join
import numpy as np
import time
from tensorflow.keras.models import load_model
from os import chdir




#Paramètres par défaut
data_path="../../../"
Mat_path="../Materiel/"
neurone_feature=Mat_path+"drop_out.50"
CNNmodel  = load_model(neurone_feature,compile=False)
path_folder=data_path+"DonneesPI/"
fichierClasses= Mat_path+"Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False) # table of species into classes 


#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv(Mat_path+"imagettes.csv")
imagettes=fn.to_reference_labels (imagettes,"classe",frame)
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 


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
tresh_list=[0.5]
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
    
        #chdir(data_path+folder)
        liste_image_ref,estimates_FP_liste,estimates_match_brids_liste,Diff_image_FP_liste,Diff_image_animals_liste,Diff_image_image_total_liste=[[] for i in range(6)]
        
    
        for r, d, f in os.walk(data_path+folder[1:]):
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







