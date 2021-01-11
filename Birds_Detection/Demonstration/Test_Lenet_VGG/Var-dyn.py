#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:35:42 2021

@author: marcpozzo
"""

import random

first_names=('John','Andy','Joe')
last_names=('Johnson','Smith','Williams')



full_name=random.choice(first_names)+" "+random.choice(last_names)

group=full_name*3


foo = "bar"
exec(foo + " = 'something else'")
print(bar)

Images["classe"].unique()

dictionnaire_gen={'corneille':0, 'ground':1, 'chevreuil':2, 'faisan':3, 'lapin':4, 'pigeon':5}
map_classes={}

for i in range(len(an_)):
    exec(an_[i]+'_'+ " = []")


an_=['corneille', 'ground', 'faisan', 'lapin', 'chevreuil', 'pigeon']
dic_dic={}
for i in range(len(an_)):
    #dic_dic[i]=exec(an_[i]+'__'+ " = []")
    exec(an_[i]+'__'+ " = []")

Tiny_images=pd.read_csv(Mat_path+"Tiny_images.csv")

#Maintenant dans ce sens 
#On donne un nom 


#Description du problème:
On a N categories provenant de l'entrainement.
On veut classer les images générées en fonction de l'imagette la plus proche correspondante.
pour cela il faut créer dynamiquement des listes (construire non à partir de string)
Il est par contre d'avantage compliqué de remplir la liste de manière dynamique



class NamedObject:
    def __init__(self, name, obj):
        self.name = name
        self.obj = obj

    def __getattr__(self, attr):
        if attr == 'name':
            return self.name
        else:
            return getattr(self.obj, attr)


unnamed_list = [1, 2, 3]
named_list = NamedObject('named_list', unnamed_list)

print(named_list) # [1, 2, 3]
print(named_list.name) # 'named_list'
