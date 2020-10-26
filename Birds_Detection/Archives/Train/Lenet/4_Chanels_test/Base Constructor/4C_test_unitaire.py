#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:14:24 2020

@author: marcpozzo
"""


from os import chdir
chdir("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/bin")
import pandas as pd
#import functions_court as fn
#import functions_netoyees as fns
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
#from tensorflow.keras.models import load_model
import pickle
import time
from tensorflow import keras

"""
nombre TP par dossier [201]
nombre FP par dossier [911]
nombre de Fp au dessus du seuil de 0.9 [595]
nombre de Tp au dessus du seuil de 0.9 [195]
"""
 

#Paramètres à choisir

#CNNmodel = tensorflow.keras.models.load_model(neurone_feature)

"""
neurone_feature="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"
neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes/Recover6"
#neurone_feature="/mnt/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob"
#neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/Iteration_Gen_Keras/10"
CNNmodel  = load_model(neurone_feature,compile=False)
#CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
"""


#neurone='/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/148ep_0.5dpt_GBR__6CL_4CH'
neurone='/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/birds_fp_matched/400ep_0.1dpt_HSV_6CL_4CH_smal_fp_birds_matched'
#neurone='/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/200ep_GBR_6CL_4CH'
neurone="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/4Chanels_6Classes/all_fp+all_birds/10ep_0.1dpt_HSV__6CL_4CH_half_fp_birds_matched"

neurone="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob"


neurone="mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/z1.1"
neurone='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes/200_ep_sans_transfo'
neurone='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes/200_ep_sans_transfo_bis'

#/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/match+fp_db/lessfp_200ep
#CNNmodel = keras.models.load_model(neurone)

neurone="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/essai_tf/tf_model"
neurone="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/correct_recall/models/match+fp_db/lessfp_200ep"
CNNmodel= tensorflow.keras.models.load_model(neurone)


path_folder="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/"


#Paramètres par défaut
path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
#frame=pd.read_csv(fichierClasses,index_col=False)





#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
#imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']


#fns.extract_fp_4C(path_images,name_test,name_ref,folder,CNNmodel,path_images)
#path_images="z"
#fns.extract_fp_4C(name_test,name_ref,folder,CNNmodel)

liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_1'  ]


Diff_image_FP_by_folder=[]
Diff_image_animals_by_folder=[]
Diff_image_image_total_by_folder=[]


start=time.time()
nb_FP_liste=[]
nb_TP_liste=[]
nb_FP_liste_thresh=[]
nb_TP_liste_thresh=[]
for folder in liste_folders:

    chdir(path+folder)
    liste_image_ref = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path+folder):
        for file in f:
            if '.jpg' in file:
                liste_image_ref.append(basename(join(r, file)))
                
    liste_image_ref.sort()           
    path_images=folder+"/"
    #folder="/DonneesPI/timeLapsePhotos_Pi1_1"
    folder_choosen="."+folder
    #imagettes_PI_0=imagettes[imagettes["path"]=="/DonneesPI/timeLapsePhotos_Pi1_2" ]

    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ]
    
    #Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer


    
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    print("nombre d'images pour le dossier",len(liste_name_test))
    
    estimates_FP_liste=[]
    estimates_match_brids_liste=[]
    folder_name=folder[-5:]
    
    
    Diff_image_FP_liste=[]
    Diff_image_animals_liste=[]
    Diff_image_image_total_liste=[]
    
    nb_folder_TP=0
    nb_folder_FP=0
    nb_TP_folder_thresh=0
    nb_FP_folder_thresh=0
    for name_test in liste_name_test:
        

        
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        

        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=fns.base_4C(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="quantile_filtre",blockSize=17,blurFact=17)
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=base_4C_poly(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="No_filtre",blockSize=17,blurFact=17,chanels=3,contrast=-8,thresh=0.9,down_thresh=25)
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=base_option(name_test,name_ref,folder,method="light")
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=base_4C_poly(name_test,name_ref,folder,CNNmodel,maxAnalDL=20,method="light",diff_mod="HSV",mask=True,filtre_choice="No_filtre",blockSize=25,blurFact=25,chanels=3,contrast=-8,thresh=0.55,down_thresh=25,seuil=210)
        imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=fn.base_4C_poly(name_test,name_ref,
                                                                                                                        folder,CNNmodel,blockSize=17,
                                                                                                                        blurFact=17,chanels=3,contrast=-8,maxAnalDL=20,method="ssim",mask=True)
        #im
        #base(name_test,name_ref,folder,CNNmodel,mask=False)
        
        
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



Avec le masque
Nouveau
nombre TP par dossier [180]
nombre FP par dossier [459]

nombre de Tp au dessus du seuil de 0.9 [178]
nombre de Fp au dessus du seuil de 0.9 [393]


Ancien
nombre TP par dossier [112]
nombre FP par dossier [430]
nombre de Tp au dessus du seuil de 0.9 [102]
nombre de Fp au dessus du seuil de 0.9 [326]

Sans le masque

Nouveau
nombre TP par dossier [200]
nombre FP par dossier [900]
nombre de Tp au dessus du seuil de 0.9 [199]
nombre de Fp au dessus du seuil de 0.9 [740]

Ancien

nombre TP par dossier [200]
nombre FP par dossier [900]

nombre de Tp au dessus du seuil de 0.9 [194]
nombre de Fp au dessus du seuil de 0.9 [589]



"""
    
"""
sudo docker-compose up
positionning.php
vim bashServer.py 
setup.md
"""