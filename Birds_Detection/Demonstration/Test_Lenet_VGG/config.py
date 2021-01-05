#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:49:56 2021

@author: marcpozzo
"""
#Penser à mettre en minuscule, le Y, message de warning si non inclu
#Importer la table
#Le remettre à l'interieur, c'est pas très beau ... 

import pandas as pd
Mat_path="../../Materiels/"
Images=pd.read_csv(Mat_path+"images.csv")


"""
liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,liste_DIFF_faisan,liste_DIFF_corbeau,liste_DIFF_pigeon,liste_DIFF_lapin,liste_DIFF_chevreuil=([] for i in range(8))

map_classes={"faisan":liste_DIFF_faisan, "corneille" : liste_DIFF_corbeau,"pigeon":liste_DIFF_pigeon,
                 "lapin" :liste_DIFF_lapin, "chevreuil" :liste_DIFF_chevreuil, "oiseau" : liste_DIFF_birds_undefined,
                 "incertain": liste_DIFF_birds_undefined, "pie":liste_DIFF_birds_undefined }
    

object_targeted=[]
object_not_targeted=[]

map_classes={"faisan":object_targeted, "corneille" : object_targeted,"pigeon":object_targeted,
                 "lapin" :object_not_targeted, "chevreuil" :object_not_targeted, "oiseau" : object_targeted,
                 "incertain": object_targeted, "pie":object_targeted }

"""

map_classes={}
object_targeted=[]
object_not_targeted=[]

OBJECTS=Images["classe"].unique()

for objects in OBJECTS:
    print(objects)
    keep=input("If you want include this objects in yours targets, please type Y")
    if keep=="Y":
        map_classes[objects]=object_targeted
    else:
        map_classes[objects]=object_not_targeted
        
        
#On demande les classes voulues
#On supprime celles qui ne correspondent pas et puis il n'y a plus qu'une classe.