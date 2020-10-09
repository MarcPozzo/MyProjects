#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:18:33 2020

@author: marcpozzo
"""

import tensorflow


from os import chdir
chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/bin")
import pandas as pd
import functions as fn
import os
from os.path import basename, join
import numpy as np
import time
from tensorflow.keras.models import load_model
import pickle

neurone_feature="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"
CNNmodel  = load_model(neurone_feature,compile=False)

path_folder="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/"
path_to_save='/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/Materiels/listes/Res_Lenet/'

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




Diff_image_FP_by_folder,Diff_image_animals_by_folder,Diff_image_image_total_by_folder,nb_FP_liste,nb_TP_liste,nb_FP_liste_thresh,nb_TP_liste_thresh=[[] for i in range(7)]


#bird_prob
liste_parameter=[("bird_large","light",25,50),("bird_large","light",25,-1),("bird_large","light",17,20)]


for parameter in liste_parameter:
    focus,method,blockSize,maxAnalDL=parameter
    base_name=focus+"-"+method+"-"+str(blockSize)+"-"+str(maxAnalDL)+".txt"
    nb_FP_liste=[]
    nb_TP_liste=[]
    
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
            imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux=fn.base_4C_poly(name_test,name_ref,
                                                                                                                            folder,CNNmodel,blockSize=blockSize,thresh=0.5,
                                                                                                                            blurFact=17,chanels=3,contrast=-8,maxAnalDL=maxAnalDL,method=method,mask=True,focus=focus)
           
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
       
        
        with open(path_to_save+"FP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_FP_liste, fp)
            
        with open(path_to_save+"TP-"+base_name, "wb") as fp:   #Pickling
            pickle.dump(nb_TP_liste, fp)
            
            
