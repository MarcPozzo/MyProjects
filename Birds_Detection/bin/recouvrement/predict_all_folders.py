#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:09:54 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-

#L'objectif de ce script est de tester de grands échantillons d'images pour trouver les meilleurs paramètres pour le filtre par différence.
#La fonction de recouvrement testé peut renvoyer la somme des oiseaux à trouver sur le script (nb_birds_to_find), les oiseaux trouvés par
#le filtre (nb_birds_match), les annotations identifiées et dont la prédiction fait partie d'une des 5 catégories d'animaux prévues par
#le RN (TP), les mottes de terre et d'herbes qui ne correspondent pas aux annotations et identifiés comme l'un des 5 animaux de la pred FP
#Les anotations dont la prédiction correspond à la bonne catégorie VTP



#Importation packages

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import pandas as pd
import functions as fn
import os
from os.path import basename, join
import numpy as np
import ast
from keras.models import Model, load_model
import cv2
            
            


#Paramètres à choisir


coverage_threshold=0.5
#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLAapsePhotos_Pi1_4'   ]
#Pour filtre quantile
neurone_features='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob'
#neurone_features='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models/z1.2_r_20_drpt_0.1'
#Para optimaux pour dossier 1
contrast=-5
blockSize=19
blurFact=15



#Paramètres par défaut
path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]
fichierClasses= "/mnt/VegaSlowDataDisk/c3po/Images_aquises/Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)








#Dictionnaire pour ensuite transformer les labels [0:5] avec le label en string qui correspond
dictionnaire_conversion={}
dictionnaire_conversion[0]="autre"
dictionnaire_conversion[1]="chevreuil"
dictionnaire_conversion[2]="corneille"
dictionnaire_conversion[3]="faisan"
dictionnaire_conversion[4]="lapin"
dictionnaire_conversion[5]="pigeon"









neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/drop_out.50"
#neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob"
#neurone_feature="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/Iteration_Gen_Keras/10"
model = load_model(neurone_feature,compile=False)
CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)







#import time


#debut = time.time()
# On attend quelques secondes avant de taper la commande suivante



#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0'] 
birds_match_by_folder=[]
Birds_well_predict_by_folder=[]
birds_predict_by_folder=[]
liste_Nb_oiseaux_a_reperer=[]
Nb_oiseaux_a_reperer_by_folder=[]
Nombre_imagette_oiseau_match_by_folder=[]
liste_FP_by_folder=[]
liste_imagettes_by_folder=[]
nb_animals_to_find_by_folder=[]
nb_animals_match_by_folder=[]
birds_defined_match_by_folder=[]


imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes=to_reference_labels (imagettes,"classe")
#L'image a été supprimé, il faudrait généré de nouveaux les images à l'occasion
imagettes=imagettes[imagettes["filename"]!='image_2019-04-18_17-56-42.jpg']
imagettes=imagettes[imagettes["classe"]!="ground"]    
imagettes=imagettes[  (imagettes["classe"]!="autre") 
                  & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                  & (imagettes["classe"]!="cheval") ]


"""
imagettes=imagettes[  (imagettes["classe"]!="oiseau") 
                  & (imagettes["classe"]!="incertain")]
"""


#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0']
for folder in liste_folders:

    """imagettes=fn.to_reference_labels (imagettes,"classe")
    imagettes=imagettes[ (imagettes["classe"]!="oiseau") & (imagettes["classe"]!="autre") & (imagettes["classe"]!="pie") 
                        & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                        & (imagettes["classe"]!="cheval") ]"""
    
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
    for name_test in liste_name_test:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        nb_animals_to_find,nb_animals_match,birds_to_find,birds_match,birds_predict,birds_defined_match,VTP,nombre_imagettes,FP=bazard2(path_images,name_test,name_ref,folder,CNNmodel,blockSize=53,blurFact=15)

        
        #nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        #nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP=fn.birds_square_light(path_images,name_test,name_ref,folder,neurone_features)
       
        #Nombre_imagette_oiseau_match.append()
        birds_defined_match_liste.append(birds_defined_match)
        liste_birds_match.append(birds_match)
        nb_animals_to_find_liste.append(nb_animals_to_find)
        nb_animals_match_liste.append(nb_animals_match)
        Nb_oiseaux_a_reperer.append(birds_to_find)
        Birds_predict.append(birds_predict)
        Birds_well_predict.append(VTP)
        liste_FP.append(FP)
        liste_imagettes.append(nombre_imagettes)
    #print(Birds_predict)
    
    nb_animals_match_by_folder.append(sum(nb_animals_match_liste))
    nb_animals_to_find_by_folder.append(sum(nb_animals_to_find_liste))
    Nb_oiseaux_a_reperer_by_folder.append(sum(Nb_oiseaux_a_reperer))
    birds_match_by_folder.append(sum(liste_birds_match))
    
    
    print(sum(Birds_predict))
    print(folder)
    birds_predict_by_folder.append(sum(Birds_predict))   
    Nombre_imagette_oiseau_match_by_folder.append(sum(Nombre_imagette_oiseau_match))
    Birds_well_predict_by_folder.append(sum(Birds_well_predict))
    liste_FP_by_folder.append(sum(liste_FP))
    liste_imagettes_by_folder.append(sum(liste_imagettes))
    birds_defined_match_by_folder.append(sum(birds_defined_match_liste))
    

print("Nb_oiseaux total a_reperer par dossier",Nb_oiseaux_a_reperer_by_folder)
print("Nombre_imagette_oiseau_match_by_folder",birds_match_by_folder)
#print("Nombre_imagette_oiseau_match_by_folder",Nombre_imagette_oiseau_match_by_folder)
print("birds_predict_by_folder",birds_predict_by_folder)

print("birds_defined_match_by_folder",birds_defined_match_by_folder)
print("Birds_well_predict by folder",Birds_well_predict_by_folder)
print("nompbre d'imagettes générées par dossier",liste_imagettes_by_folder)
print("le nombre de faux positifs par dossier est",liste_FP_by_folder)


#fin = time.time()

#print("le programme a mis pour tourner",fin-debut)

#15h40




#essayer de trouver les anciens para et enlever les indefinis 


#nb_animals_to_find,nb_animals_match,pourcentage_FP,animals_predict,FP,VTP,birds_predict


#Plus bas si on veut faire des compilations par paramètres



"""




"""

"""

Nb_oiseaux_a_reperer_by_folder [29, 29, 29, 29]
Nombre_imagette_oiseau_match_by_folder [19, 19, 19, 19]
birds_predict_by_folder [16, 16, 16, 16]
Birds_well_predict by folder [8, 8, 8, 8]
blocSize_liste=[3,201,501]







#Para par défaut
contrast=-5
blocSize_liste=[19]
#blurFact=15
blurFact=7


blurFact_liste=[15,7,9,11,13,17,25,33]
contrast_liste=[-5,-15,-10,5,10,15]
blocSize_liste=[19,7,9,11,13,15]


"""

"""
blurFact_liste=15
blurFact=7
#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0'] 
Birds_well_predict_by_folder=[]




neurone_path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/"
neurone_features="zoom_models/z1.3"
neurone_dir1=os.listdir("/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models")
neurone_dir2=os.listdir("/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models")
neurone_dir=neurone_dir1+neurone_dir2
neurone_dir1.remove('.ipynb_checkpoints')
neurone_path_iteration="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models/"

neurone_path_zoom="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/"



Birds_well_predict_by_folder=[]
birds_predict_by_folder=[]
liste_Nb_oiseaux_a_reperer=[]
Nb_oiseaux_a_reperer_by_folder=[]
Nombre_imagette_oiseau_match_by_folder=[]


for nn in neurone_dir2:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    neurone_features=neurone_path_zoom+nn






    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=fn.to_reference_labels (imagettes,"classe")
    imagettes=imagettes[ (imagettes["classe"]!="oiseau") & (imagettes["classe"]!="autre") & (imagettes["classe"]!="pie") 
                        & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                        & (imagettes["classe"]!="cheval") ]
    


    imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]



    Birds_well_predict=[]
    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[:19]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        #nombre_imagette_oiseau_match,pourcentage_TP,nb_oiseaux_a_reperer,TP=birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        
        nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        
        Pourcentage_birds_predict.append(pourcentage_TP)
        Nombre_imagette_oiseau_match.append(nb_birds_match)
        Nb_oiseaux_a_reperer.append(nb_birds_to_find)
        Birds_predict.append(TP)
        Birds_well_predict.append(VTP)
    #print(Birds_predict)
    Nb_oiseaux_a_reperer_by_folder.append(sum(Nb_oiseaux_a_reperer))
    print(sum(Birds_predict))
    print(folder)
    birds_predict_by_folder.append(sum(Birds_predict))   
    Nombre_imagette_oiseau_match_by_folder.append(sum(Nombre_imagette_oiseau_match))
    Birds_well_predict_by_folder.append(sum(Birds_well_predict))

print("Nb_oiseaux_a_reperer_by_folder",Nb_oiseaux_a_reperer_by_folder)
print("Nombre_imagette_oiseau_match_by_folder",Nombre_imagette_oiseau_match_by_folder)
print("birds_predict_by_folder",birds_predict_by_folder)
print("Birds_well_predict by folder",Birds_well_predict_by_folder)



#Le numéro 20 21 et 31 oiseaux de différents types

#Deuxième série de neurones


birds_predict_by_folder [13, 10, 7, 11, 11, 9, 9]
Birds_well_predict by folder [7, 8, 7, 7, 7, 6, 8, 8, 8, 4, 5, 5, 5, 6, 5]


Nb_oiseaux_a_reperer_by_folder [29, 29, 29, 29, 29, 29, 29]
Nombre_imagette_oiseau_match_by_folder [18, 18, 18, 18, 18, 18, 18]
birds_predict_by_folder [13, 10, 7, 11, 11, 9, 9]
Birds_well_predict by folder [7, 8, 7, 7, 7, 6, 8, 8, 8, 4, 5, 5, 5, 6, 5]

for nn in ['z1.2_r_20_drpt_0.1']:

    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    neurone_features=neurone_path_iteration+nn





    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    imagettes=fn.to_reference_labels (imagettes,"classe")
    imagettes=imagettes[ (imagettes["classe"]!="oiseau") & (imagettes["classe"]!="autre") & (imagettes["classe"]!="pie") 
                        & (imagettes["classe"]!="chat") & (imagettes["classe"]!="abeille") & (imagettes["classe"]!="sanglier") 
                        & (imagettes["classe"]!="cheval") ]
    


    imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]



    Birds_well_predict=[]
    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[:19]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        #nombre_imagette_oiseau_match,pourcentage_TP,nb_oiseaux_a_reperer,TP=birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        
        nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        
        Pourcentage_birds_predict.append(pourcentage_TP)
        Nombre_imagette_oiseau_match.append(nb_birds_match)
        Nb_oiseaux_a_reperer.append(nb_birds_to_find)
        Birds_predict.append(TP)
        Birds_well_predict.append(VTP)
    #print(Birds_predict)
    Nb_oiseaux_a_reperer_by_folder.append(sum(Nb_oiseaux_a_reperer))
    print(sum(Birds_predict))
    print(folder)
    birds_predict_by_folder.append(sum(Birds_predict))   
    Nombre_imagette_oiseau_match_by_folder.append(sum(Nombre_imagette_oiseau_match))
    Birds_well_predict_by_folder.append(sum(Birds_well_predict))

print("Nb_oiseaux_a_reperer_by_folder",Nb_oiseaux_a_reperer_by_folder)
print("Nombre_imagette_oiseau_match_by_folder",Nombre_imagette_oiseau_match_by_folder)
print("birds_predict_by_folder",birds_predict_by_folder)
print("Birds_well_predict by folder",Birds_well_predict_by_folder)




        debut = time.time()
        model = load_model(neurone_features,compile=False)
        CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
        fin=time.time()
        print("importation fini", fin-debut)
        
#debut = time.time()
fn.birds_precis(path_images,name_test,name_ref,folder,CNNmodel)

#fin = time.time()

#print("cela dure", fin-debut)


name_test,name_ref="image_2019-06-15_04-20-05.jpg", "image_2019-06-15_04-19-49.jpg"


name_test,name_ref="image_2019-06-14_16-50-21.jpg", "image_2019-06-14_16-50-04.jpg"
















#fn.birds_square_light(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)


"""