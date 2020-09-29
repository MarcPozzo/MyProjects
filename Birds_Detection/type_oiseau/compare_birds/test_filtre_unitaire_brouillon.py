#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:28 2020

@author: pi




"""
#L'objectif de ce script est d'établir des bons paramètres pour le filtre.
#Par la suite il pourrait être judicieux de trouver les meilleurs paramètres en évaluant le pourcentage le nombre de bonnes annotations
#recouvertes, mais aussi en essayant de minimiser de le nombre de carrés superflux dans un deuxième temps (moins  important)


#Maintenant le sous objectif est de rajouter la fonction is-catched mais de manière assez facile à utiliser
from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
#import functions as fn
exec(open("functions.py").read())
import cv2
import pandas as pd
import joblib
import matplotlib.pyplot as plt

output_Images_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/"

#Paramètres à choisir

#Pour filtre quantile



filtre_choice="No_filtre" #"quantile_filtre"#"No_filtre" "RL_filtre"



#Autres Paramètres
path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/6classes_zoom/"







#Avec le même répertoire 



path_anotation="testingInputs/oiseau_lab_Alex.csv"



#Avec les photos du pi

#Dans le dossier1 du PI Ceux la moitié fonctionne


path_image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-27-49.jpg"

path_image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-24-28.jpg"








def filtre_light(imageA,imageB,contrast=-5):
    img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    blurFact = 11
    absDiff2 = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(src=diff,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                                thresholdType=cv2.THRESH_BINARY,blockSize=221,C=contrast) # adaptation de C à histogram de la photo ?

    th2Blur=cv2.GaussianBlur(th2,(blurFact,blurFact),sigmaX=0)
    th2BlurTh = cv2.adaptiveThreshold(src=th2Blur,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
            thresholdType=cv2.THRESH_BINARY,blockSize=121,C=contrast) # adaptation de C à histogram de la photo ?
    threshS=th2BlurTh

        # defines corresponding regions of change
    cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    return cnts


imageA = cv2.imread(path_image_ref)
imageB = cv2.imread(path_image_test)

cv2.imread(image_ref)
cnts=filtre_light(imageA,imageB)
print(len(cnts))

place_generate_sqaure(imageB,cnts)




birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2,name_test,height=2448,width=3264)


#Dans le dossier 3
path_image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_07-57-16.jpg"
path_image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_07-56-59.jpg"

imageA = cv2.imread(path_image_ref)
imageB = cv2.imread(path_image_test)

cnts=filtre_light(imageA,imageB)
print(len(cnts))




place_generate_sqaure(imageB,cnts)
birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2,name_test,height=2448,width=3264)











place_generate_sqaure(imageB,cnts)
birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,name2,name_test)


#Le mieux ce serait namee_test, name_ref et path_imagettes donc plus de image A, B name2 #path annotation est devenu inutile 
#Peut être que dans un premier temps on peut mettre en paramètre par défaut neurone_features, filtre_choice, coef_filtre



path_images="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"

name_test = "image_2019-04-30_18-25-35.jpg"
name_ref="image_2019-04-30_18-24-28.jpg"
#birds_is_catche(image_ref,image_test,name_test,name_ref)


imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes_PI_0=imagettes[(imagettes["path"]=="./DonneesPI/timeLapsePhotos_Pi1_0") ]

for i in range(len(imagettes_PI_0)):
    name_test=imagettes_PI_0["filename"].iloc[i]
    print(name_test)
    birds_is_catche(path_images,name_test,name_ref)


#Il faudrait s'assurer qu'on ne prenne pas en compte les imagettes de terre
#dict_images_catched={}
nb_of_images_catched={}
for i in range(len(imagettes_PI_0)):
    name_test=imagettes_PI_0["filename"].iloc[i]
    print(name_test)
    birds_is_catche(path_images,name_test,name_ref)
    nb_of_images_catched[name_test]=birds_is_catche
    
    
#On va faire le test en prennant l'image précédante

#Il semblerait que le problème c'est que les images soient tout simplement les mêmes
#Si je fais la méthode de Corentin en prenant l'image juste avant c tomber sur ce genre de pb donc faudrait un debug auto.
    
    
#A priori le dictionnaire va être fait donc on peut maintenant faire un programme pour vérifier 
#Que les imagettes sont bien trouvées selon deux manières
#Premièrement combien on a identifié d'images pour lesquelles il y a toutes les imagettes
#Deuxièmement quel est le pourcentage d'imagette on a repéré (facile le nombre de ligne totale divisé par 2 le nombre additionnée des vals du dic)
    

image_reussies=0
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test:
    print(name_test)
    if nb_of_images_catched[name_test]==len(imagettes_PI_0[imagettes_PI_0["filename"]==name_test]):
        image_reussies+=1
        
print("le nombre d'images sur lesquelles toutes les imagettes sont identifiées est",image_reussies)

print("le pourcentage d'images sur lesquelles toutes les imagettes sont identifiées est",image_reussies)

nb_catched_imagettes=len(nb_of_images_catched)
nb_imagettes_oiseaux=len(imagettes_PI_0)/2

print("le pourcentage d'imagettes extraites parmis les imagettes d'oiseau est de : ",nb_catched_imagettes/nb_imagettes_oiseaux )




#Le code fait des répitions pour rien s'il y a plusieurs imagettes
dict_images_catched={}
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test:
    print(name_test)
    catched_bird=birds_is_catche(path_images,name_test,name_ref)
    dict_images_catched[name_test]=catched_bird
    
len(dict_images_catched)

dict_images_catched

dict_images_catched={}
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test[-5:-1]:
    print(name_test)
    catched_bird=birds_is_catche(path_images,name_test,name_ref)
    dict_images_catched[name_test]=catched_bird
    
len(dict_images_catched)

dict_images_catched






name_test="image_2019-04-30_18-18-21.jpg"
birds_is_catches(path_images,name_test,name_ref)




image_reussie=0
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test:
    print(name_test)
    if dict_images_catched[name_test]==len(imagettes_PI_0[imagettes_PI_0["filename"]==name_test])/2:
        image_reussie+=1
        
nb_catched_imagettes=len(dict_images_catched)
nb_imagettes_oiseaux=len(imagettes_PI_0)/2

print("le pourcentage d'imagettes extraites parmis les imagettes d'oiseau est de : ",nb_catched_imagettes/nb_imagettes_oiseaux )

#Il doit y avoir de nombreux problème
#Déjà le nombre d'imagette théorique est non nulle il faudrait juste faire un filtre en enlevant ground
#Je n'ai comptabilisé non pas les imagettes du dic mais seulement les clefs il faudrait une somme
#Pour image réussie c'est normale il y a celle avec de la terre

dict_images_catched["image_2019-04-30_18-46-43.jpg"]