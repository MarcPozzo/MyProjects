#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:33:29 2021

@author: marcpozzo
"""
#When it comes to analyse small objects in the a big picture, the first step is to find a method to extract these object before to analyze them.
#One way to extract these object is to compare couple of images and find differences between them. Tiny images where the biggest differences appeared are generated. 
#Objects targeted should be include inside of these tiny images.
#This script allows to find the method having the best ratio between the objects extracted and the generated images.


#Import librairies
import pandas as pd
import functions_Lenet_VGG as fn






#Default Parameters
data_path='../../../../Pic_dataset/'
Mat_path="../../Materiels/"
Images=pd.read_csv(Mat_path+"images.csv")




#Please change parameters correponding to your data sets
objects_targeted_=["corneille","pigeon","faisan","oiseau","pie","incertain"] #Corresponded to the classes of objects you want to keep
liste_parameter=[(17,"light",25,-5),(15,"ssim",25,-1),(15,"light",17,5)] #Corresponded to the parameters used by the difference in the loop 


#Gather the names of picture with targeted birds
Images=Images[Images["classe"].isin(objects_targeted_)]
images_birds_=list(Images["filename"].unique()) # only images containig birds


#Gather and sort the names of all picture (with and without bird ) in a list
images_=fn.order_images(data_path)



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
        an_caught,nb_objects_to_caught,TINY_IMAGES_GENERATED=fn.Evaluate_extraction ( Images , name_test , name_ref ,data_path ,objects_targeted_, contrast, blockSize, blurFact, diff_mod3C  )
        AN_CAUGHT+=an_caught
        print("For the picture: ",name_test )
        print("Number of objects caught",an_caught)
        print("Number of objects in the picture",nb_objects_to_caught)
        print("Number of tiny images generated",TINY_IMAGES_GENERATED)
        NB_OBJECTS_TO_CAUGHT+=nb_objects_to_caught
    print("pour ce jeu de paramètre voici les résultat obtenus")
    print("nombre d'animaux détectés",AN_CAUGHT)
    print("nombre d'animaux à détecter",NB_OBJECTS_TO_CAUGHT)
    AN_CAUGHT_param.append(AN_CAUGHT)
    
        
        
        