#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""

#dropout50
from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/bin")
import pandas as pd
import functions as fn
#import functions as fns
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




#Paramètres à choisir

path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/Alex_db/tf_fp_100ep"
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/match+fp_db/tf_fp_200_ep"
path_neurone="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/match+fp_db/"


neurone="lessfp_200ep"
neurone_feature=path_neurone+neurone


#neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/train_models/55/gen"
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/train_models/50/min_fp"
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/train_models/55/test.model"
neurone_feature='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/train_models/55/0.99'
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/train_models/50/m_min_fp"
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"


neurone_feature='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/classe_fp/8c_20'
CNNmodel = tensorflow.keras.models.load_model(neurone_feature)

coverage_threshold=0.5


#Paramètres par défaut
path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]

fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0' ]



#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes=to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 

imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']

#854
#vs 668
birds_pr_birds_thresh_total_by_folder=[]
FP_birds_thresh_total_by_folder=[]
birds_predict_total=0
#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0']

#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0']
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

    liste_birds_match=[]
    nb_animals_match_liste=[]
    nb_animals_to_find_liste=[]
    Birds_well_predict=[]
    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    liste_FP=[]
    liste_imagettes=[]
    birds_defined_match_liste=[]
    dict_images_catched={}
    

    
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    estimates_FP_liste=[]
    estimates_match_brids_liste=[]
    folder_name=folder[-5:]
    (FP_birds_thresh_total,birds_pr_birds_thresh_total)=(0,0)
    for name_test in liste_name_test:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        
        #estimates_FP,estimates_match_brids,birds_predict=fns.threshold_fp(path_images,name_test,name_ref,folder,CNNmodel,blockSize=53,blurFact=15)
        #estimates_match_brids_liste+=estimates_match_brids
        #estimates_FP_liste+=estimates_FP
        #birds_predict_total+=birds_predict
        #draw_fp(path_images,name_test,name_ref,folder,CNNmodel,thresh=True,filtre_choice="quantile_filtre",coverage_threshold=0.01)
        
        
        #(name_test,name_ref)=('image_2019-04-06_07-01-25.jpg', 'image_2019-04-06_07-01-08.jpg') P2
        #ces photos ne fonctionnent pas
        #FP_birds_thresh,birds_pr_birds_thresh=fn.predict_under_thres(path_images,name_test,name_ref,folder,CNNmodel,thresh=0.9,filtre_choice="No_filtre",coverage_threshold=0.01)
        ln_FP_birds_thresh,ln_birds_pr_birds_thresh=predict_8classes(name_test,name_ref,folder,CNNmodel,numb_classes=6,mask=True,thresh=0.9,filtre_choice="No_filtre",coverage_threshold=0.01,thresh_active=True)
        
        print("///")                                           
        print("birds_pr_birds_thresh",ln_birds_pr_birds_thresh)
        print("//")
        FP_birds_thresh_total+=ln_FP_birds_thresh
        birds_pr_birds_thresh_total+=ln_birds_pr_birds_thresh
        
        
       #(name_test,name_ref)= ("image_2019-06-15_17-39-45.jpg", "image_2019-06-15_17-39-28.jpg")
    birds_pr_birds_thresh_total_by_folder.append(birds_pr_birds_thresh_total)
    FP_birds_thresh_total_by_folder.append(FP_birds_thresh_total)
    #(name_test,name_ref)= ('image_2019-06-14_15-47-11.jpg', 'image_2019-06-14_15-46-54.jpg')
    
print("nombre d'oiseaux identifiés :",birds_pr_birds_thresh_total_by_folder)
print("nombre de faux positifs", FP_birds_thresh_total_by_folder)


"""
nombre d'oiseaux identifiés : [190]
nombre de faux positifs [663]

nombre d'oiseaux identifiés : [56, 56, 140, 12, 70, 42]
nombre de faux positifs [86, 86, 451, 42, 1222, 353]


nombre d'oiseaux identifiés : [125, 303, 2, 27, 15]
nombre de faux positifs [18, 83, 0, 67, 29]
"""

random_image_path="/home/marcpozzo/Desktop/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-55-20.jpg"
random_image=cv2.imread(random_image_path)

#mask_download=load('mask_0.npy')
out_arr_1 = np.multiply(mask, random_image) 

#Image initiale
plt.figure(figsize=(25,25))
plt.subplot(1,4,1)
plt.imshow(random_image)

#Masque seul
plt.subplot(1,4,2)
plt.imshow(mask3D)

#Image masquée
plt.subplot(1,4,3)
out_arr= np.multiply(mask, random_image) 
out_arr=out_arr.astype(int)
random_image=random_image.astype(int)
plt.imshow(out_arr )