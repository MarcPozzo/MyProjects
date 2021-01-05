#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:33:29 2021

@author: marcpozzo
"""

import pandas as pd
import os
from os.path import basename, join
import functions_Lenet_VGG as fn
#from os import chdir





#Paramètres par défaut
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"


Images=pd.read_csv(Mat_path+"images.csv")
objects_targeted_=["corneille","pigeon","faisan","oiseau","pie","incertain"]
Images=Images[Images["classe"].isin(objects_targeted_)]
liste_parameter=[(17,"light",25,-5),(15,"ssim",25,-1),(15,"light",17,5)]






#Gather the names of picture with birds
images_birds_=list(Images["filename"].unique()) # only images containig birds


#Gather and sort the names of all picture (with and without bird ) in a list
images_=[]
for r, d, f in os.walk(data_path):
    for file in f:
        if '.jpg' in file:
            images_.append(basename(join(r, file)))                       
images_.sort()   


AN_CAUGHT_param=[]


#loop apply in a range of pic and in a range of pictures to evaluate the number of True Positif and False Positf and save results in list
for parameter in liste_parameter:
    AN_CAUGHT,NB_OBJECTS_TO_CAUGHT=(0,0)
    blurFact,diff_mod3C,blockSize,contrast=parameter
    #base_name=blurFact+"-"+diff_mod3C+"-"+str(blockSize)+"-"+str(contrast)+".txt"

    
     
    for name_test in images_birds_:
                       
        index_of_ref=images_.index(name_test)-1
        name_ref=images_[index_of_ref]
        print(name_test,name_ref)    
        an_caught,nb_objects_to_caught=fn.Evaluate_extraction ( Images , name_test , name_ref ,data_path ,objects_targeted_, contrast, blockSize, blurFact, diff_mod3C  )
        AN_CAUGHT+=an_caught
        print("pour cette image")
        print("Nombre d'animaux capturés",an_caught)
        print("Nombre d'animaux à capturer",nb_objects_to_caught)
        NB_OBJECTS_TO_CAUGHT+=nb_objects_to_caught
    print("pour ce jeu de paramètre voici les résultat obtenus")
    print("nombre d'animaux détectés",AN_CAUGHT)
    print("nombre d'animaux à détecter",NB_OBJECTS_TO_CAUGHT)
    AN_CAUGHT_param.append(AN_CAUGHT)
    
        
        
        