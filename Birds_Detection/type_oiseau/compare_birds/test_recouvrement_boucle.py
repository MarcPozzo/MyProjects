#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:09:54 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-

#L'objectif de ce script est d'établir des bons paramètres pour le filtre.
#Par la suite il pourrait être judicieux de trouver les meilleurs paramètres en évaluant le pourcentage le nombre de bonnes annotations
#recouvertes, mais aussi en essayant de minimiser de le nombre de carrés superflux dans un deuxième temps (moins  important)

#Dans un premier temps on va comparer les images deux par deux puis toutes les comparer à la fois à une unique ref

#Maintenant le sous objectif est de rajouter la fonction is-catched mais de manière assez facile à utiliser


#Importation packages

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import pandas as pd
import functions as fn


#Paramètres à choisir

name_ref="image_2019-04-30_17-47-10.jpg"
coverage_threshold=0.8

#Pour filtre quantile


"""
Pour coverage_threshold=0.1
le nombre d'images sur lesquelles toutes les imagettes sont identifiées est 91
le pourcentage d'image sur laquelle l'extraction se fait coorectement est :  0.8921568627450981
le pourcentage d'imagettes extraites parmi les imagettes d'oiseau est de :  0.9527272727272728
"""



"""
Pour coverage_threshold=0.8
le nombre d'images sur lesquelles toutes les imagettes sont identifiées est 80
le pourcentage d'image sur laquelle l'extraction se fait coorectement est :  0.7843137254901961
le pourcentage d'imagettes extraites parmi les imagettes d'oiseau est de :  0.9054545454545454
"""





path_images="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes_PI_0=imagettes[(imagettes["path"]=="./DonneesPI/timeLapsePhotos_Pi1_0") ]

#Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer
imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]









#Le code fait des répitions pour rien s'il y a plusieurs imagettes
dict_images_catched={}
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test:
    print(name_test)
    catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold)
    dict_images_catched[name_test]=catched_bird
    
len(dict_images_catched)





image_reussie=0
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test:
    print(name_test)
    if dict_images_catched[name_test]==len(imagettes_PI_0[imagettes_PI_0["filename"]==name_test]):
        image_reussie+=1


#Attention ici ça fonctionne car on a enlevé ground

nb_imagettes_oiseaux=len(imagettes_PI_0) 
nb_catched_imagettes=sum(dict_images_catched.values())
nb_images_oiseaux=len(liste_name_test)


print("le nombre d'images sur lesquelles toutes les imagettes sont identifiées est",image_reussie)
print("le pourcentage d'image sur laquelle l'extraction se fait coorectement est : ",image_reussie/nb_images_oiseaux )
print("le pourcentage d'imagettes extraites parmi les imagettes d'oiseau est de : ",nb_catched_imagettes/nb_imagettes_oiseaux)





liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test[0]:
    print(name_test)
    estimates,nb_birds=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold)
    estimates
import functions as fn  
coverage_threshold=0.1
#name_test=liste_name_test[0]
name_test="image_2019-04-30_18-25-35.jpg"
estimates,nb_birds=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,limit_area_square=70)
estimates,nb_birds=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,limit_area_square=70000000000)
estimates



imagettes_PI_0[imagettes_PI_0["filename"]==name_test]

liste_prediction=list(estimates.argmax(axis=1))
pourcentage_birds_predict=(len(liste_prediction)-liste_prediction.count(0))/len(liste_prediction)


