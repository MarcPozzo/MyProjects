#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:04:54 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:34:14 2020

@author: marcpozzo
"""

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
import functions_ as fns
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


 

#Paramètres à choisir

#CNNmodel = tensorflow.keras.models.load_model(neurone_feature)


neurone_feature="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"


#neurone_feature="/mnt/VegaSlowDataDisk/Backups/VegaFastExtension/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes/drop_out.50"

neurone_feature="/home/marcpozzo/Desktop/c3po_all/c3po/codePython/drop_out.50"


CNNmodel  = load_model(neurone_feature,compile=False)


#CNNmodel = keras.models.load_model(neurone)

path_folder="mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/DonneesPI/"


#Paramètres par défaut
path="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises"
fichierClasses= "/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
#frame=pd.read_csv(fichierClasses,index_col=False)





#Select only animals categories
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
#imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)] 
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["filename"]!='image_2019-04-30_18-17-14.jpg']



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
too=0
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
    
    nb_folder_TP=0
    nb_folder_FP=0
    nb_TP_folder_thresh=0
    nb_FP_folder_thresh=0
    
    
neuron_liste=["/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes/drop_out.50",
"/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"]
for n in neuron_liste:    
    
    Diff_image_FP_liste=[]
    Diff_image_animals_liste=[]
    Diff_image_image_total_liste=[]
    
    nb_folder_TP=0
    nb_folder_FP=0
    nb_TP_folder_thresh=0
    nb_FP_folder_thresh=0
    CNNmodel  = load_model(n,compile=False)
    
    for name_test in liste_name_test:
        

        
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        

        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=fns.base_4C(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="quantile_filtre",blockSize=17,blurFact=17)
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates=base_4C_poly_bis(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="No_filtre",blockSize=17,blurFact=17,chanels=3,contrast=-8,thresh=0.9,down_thresh=25,maxAnalDL=-1)
        
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,birds_defined_match,a=base_option(name_test,name_ref,folder,CNNmodel)
        #base(name_test,name_ref,folder,CNNmodel,mask=False)
        
        #imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,nb_imagettes=base_4C_poly_bis(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="No_filtre",blockSize=17,blurFact=17,chanels=3,contrast=-8,thresh=0.9,down_thresh=25,maxAnalDL=50,method="ssim")
        imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,nb_imagettes=base_4C_poly_bis(name_test,name_ref,folder,CNNmodel,diff_mod="HSV",mask=True,filtre_choice="No_filtre",blurFact=17,chanels=3,contrast=-8,thresh=0.9,blockSize=25,down_thresh=25,maxAnalDL=-1,method="ssim")
   
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

1+1


"""
nombre TP par dossier [103]
nombre FP par dossier [1125]
nombre de Tp au dessus du seuil de 0.9 [63]
nombre de Fp au dessus du seuil de 0.9 [261]




nombre TP par dossier [105]
nombre FP par dossier [585]
nombre de Tp au dessus du seuil de 0.9 [63]
nombre de Fp au dessus du seuil de 0.9 [192]
"""
