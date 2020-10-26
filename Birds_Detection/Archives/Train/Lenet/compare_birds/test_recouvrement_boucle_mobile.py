#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:09:54 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-

#L'objectif est d'implémenter une référence mobile, d'abord en prenant en compte les imagettes annotées
#puis en trouvant une solution pour que ça soit aussi les non annotées car il reste le même problème de l'ombre

#Importation packages

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import pandas as pd
import functions as fn
import os
from os.path import basename, join
import numpy as np

chdir("/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/")
liste_image_ref = []
# r=root, d=directories, f = files
for r, d, f in os.walk("/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"):
    for file in f:
        if '.jpg' in file:
            liste_image_ref.append(basename(join(r, file)))
            
            


#Paramètres à choisir

name_ref="image_2019-04-30_17-47-10.jpg"
coverage_threshold=0.5

#Pour filtre quantile







path_images="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes_PI_0=imagettes[(imagettes["path"]=="./DonneesPI/timeLapsePhotos_Pi1_0") ]

#Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer
imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]


#Les resultats semblent bons.
# 2 choses à faire aller voir la taille des annotations
#Aller voir les estimations à la riguer dire le pourcentage qui n'est pas autre et le printer
#diff entre nb_oiseaux_a_reperer ça s'est via carre annnotation et 







###############Maintenant on s'occupe des seuilles de couveraes
coverage_threshold=[0.5]
Nb_oiseaux_a_reperer=[]
Nombre_imagette_oiseau_match=[]
Pourcentage_birds_predict=[]
Birds_predict=[]




dict_images_catched={}
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test[0:30]:
    index_of_ref=liste_image_ref.index(name_test)-1
    name_ref=liste_image_ref[index_of_ref]
    print(name_test,name_ref)
    #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
    #dict_images_catched[name_test]=catched_bird
    nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
    Pourcentage_birds_predict.append(pourcentage_birds_predict)
    Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
    Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
    Birds_predict.append(birds_predict)
len(dict_images_catched)


sum(Birds_predict)





birds_predict_by_coverage=[]




liste_coverage_threshold=np.arange(0.1,0.9,0.1)





for coverage_threshold in liste_coverage_threshold:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:30]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        Pourcentage_birds_predict.append(pourcentage_birds_predict)
        Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
        Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
        Birds_predict.append(birds_predict)

    print(Birds_predict)
    print(sum(Birds_predict))
    print(coverage_threshold)
    birds_predict_by_coverage.append(sum(Birds_predict))







#54 56 oiseaux et 32 trouvées 
import functions as fn
contrast=-5
nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast)




estimates,nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict







""" image à tester:
    image_2019-04-30_18-18-21.jpg 
    image_2019-04-30_18-18-04.jpg """


###############Maintenant on s'occupe du contrast
birds_predict_by_contrast=[]



coverage_threshold=0.5
contrast=-5





for contrast in liste_contrast:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:30]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features)
        Pourcentage_birds_predict.append(pourcentage_birds_predict)
        Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
        Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
        Birds_predict.append(birds_predict)

    print(Birds_predict)
    print(sum(Birds_predict))
    print(contrast)
    birds_predict_by_contrast.append(sum(Birds_predict))



"""
#Resultat 
birds_predict_by_contrast=[24, 32, 32, 2, 0, 0, 0, 0, 0] entre -15 et 25
"""



######Réseaux de neurones





birds_predict_by_nn_zoom=[]
birds_predict_by_nn_iteration=[]
coverage_threshold=0.5
contrast=-5
neurone_path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/"
neurone_features="zoom_models/z1.3"
neurone_dir1=os.listdir("/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models")
neurone_dir2=os.listdir("/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models")
neurone_dir=neurone_dir1+neurone_dir2
neurone_dir1.remove('.ipynb_checkpoints')
neurone_path_iteration="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models/"

neurone_path_zoom="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/"

for nn in neurone_dir2:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    neurone_features=neurone_path_zoom+nn
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:30]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features)
        Pourcentage_birds_predict.append(pourcentage_birds_predict)
        Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
        Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
        Birds_predict.append(birds_predict)

    print(Birds_predict)
    print(sum(Birds_predict))
    print(neurone_features)
    birds_predict_by_nn_zoom.append(sum(Birds_predict))


#Résultat
"""
Oiseaux trouvés parmis
Etudié parmi ['6c_rob','drop_out.50','z1.1','z1.2','z1.3','zoom_0.9:1.3_flip','zoom_1.3']
[34, 26, 25, 32, 32, 30, 32]
"""
#Il pourrait être intéressant de refaire tourner un modèle sur les imagettes larger mais sans zoom (avec tf et keras)

#A faire demain
for nn in neurone_dir1[2:]:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    neurone_features=neurone_path_iteration+nn
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:30]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features)
        Pourcentage_birds_predict.append(pourcentage_birds_predict)
        Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
        Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
        Birds_predict.append(birds_predict)

    print(Birds_predict)
    print(sum(Birds_predict))
    print(neurone_features)
    birds_predict_by_nn_iteration.append(sum(Birds_predict))



"""
#Oiseaux trouvé parmis
['iteration_no_transformation','z1.2_r_20','z1.2_r_20_drpt_0.1','z1.2_r_20_hflip_rotation_10']

[29, 34,35,34]
"""


#Enregistrer functions
#####On va teste blocksize maintenant
#Entre 11 et 101
    
#blocksize_liste=np.arange(11,102,10)
blocksize_liste=np.arange(15,25,2)
birds_predict_by_blocksize=[]
coverage_threshold=0.5
contrast=-5
neurone_features="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models/z1.2_r_20_drpt_0.1"




for blockSize in blocksize_liste:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:30]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize)
        Pourcentage_birds_predict.append(pourcentage_birds_predict)
        Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
        Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
        Birds_predict.append(birds_predict)

    print(Birds_predict)
    print(sum(Birds_predict))
    print(neurone_features)
    birds_predict_by_blocksize.append(sum(Birds_predict))    
    
"""  
#A 11 on obtient 41 oiseaux
Resultat en fonction de la taille du block size
taille testée et résultat ci-dessous
[0,19,36, 42, 43, 46,          44, 41, 36, 35, 31, 30]
[7,9, 11, 15, 17, 19          21, 31, 41, 51, 61, 71]
"""
  
15, 17, 19,21 , 23, 25
###On pourra maintenant s'occuper du blueFact en rajoutant ahssj le bon paramètre pour nlock_size




blurFact_liste=np.arange(15,32,2)
birds_predict_by_blurFact=[]
coverage_threshold=0.5
contrast=-5
neurone_features="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob"
blockSize=blockSize 



for blurFact in blurFact_liste:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:30]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact)
        Pourcentage_birds_predict.append(pourcentage_birds_predict)
        Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
        Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
        Birds_predict.append(birds_predict)

    print(Birds_predict)
    print(sum(Birds_predict))
    print(neurone_features)
    birds_predict_by_blurFact.append(sum(Birds_predict))   


"""Resultat en fonction de blurFact
[15, 17, 19, 21, 25
[50, 50, 49, 48, 46]
"""

blurFact=15
fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact)



#Ce qui serair bien que je fasse maintenant c'est d'aller voir pour un filtre de moins 5 et un recouvrement de 0.7 d'aller voir les photos qui n'ont pas fonctionné
#Est ce que les différences sont trop grandes 
#Dans ce cas éventuellement appliquer une restiction de taille

#Ensuite on pourra ensuite faire la taille de block size

#Dans ce cas aller peut etre dans opti_para filtre.





coverage_threshold=0.5
contrast=-5






Nb_oiseaux_a_reperer=[]
Nombre_imagette_oiseau_match=[]
Pourcentage_birds_predict=[]
Birds_predict=[]
neurone_features="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/z1.3"
#Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
dict_images_catched={}
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test[0:2]:
    index_of_ref=liste_image_ref.index(name_test)-1
    name_ref=liste_image_ref[index_of_ref]
    print(name_test,name_ref)
    #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
    #dict_images_catched[name_test]=catched_bird
    nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features)
    Pourcentage_birds_predict.append(pourcentage_birds_predict)
    Nombre_imagette_oiseau_match.append(nombre_imagette_oiseau_match)
    Nb_oiseaux_a_reperer.append(nb_oiseaux_a_reperer)
    Birds_predict.append(birds_predict)

print("Nb_oiseaux_predits",Birds_predict)
print(sum(Birds_predict))
print("Nb_oiseaux_a_reperer",Nb_oiseaux_a_reperer)
print("Nombre_imagette_oiseau_match",Nombre_imagette_oiseau_match)
print("Pourcentage_birds_predict",Pourcentage_birds_predict)



#Meilleur résultat

"""
Nb_oiseaux_predits [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 4, 5, 5, 5, 3]
32
Nb_oiseaux_a_reperer [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 7, 7, 6, 5, 4]
Nombre_imagette_oiseau_match [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 6, 7, 6, 5, 4]
Pourcentage_birds_predict [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.6666666666666666, 0.7142857142857143, 0.8333333333333334, 1.0, 0.75]
"""

#Quelles sont les images que je vais électionner
liste_name_test
Nb_oiseaux_predits [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 4, 5, 5, 5, 3]
Nombre_imagette_oiseau_match [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 6, 7, 6, 5, 4]
