#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:38:57 2020

@author: marcpozzo
"""

imagettes_size=imagettes.copy()
largeur=imagettes_size["xmax"]-imagettes_size["xmin"]
longueur=imagettes_size["ymax"]-imagettes_size["ymin"]
imagettes_size["largeur"]=largeur
imagettes_size["longueur"]=longueur

plt.hist(longueur/largeur)

liste_birds=["corneille","pigeon","faisan"]

for i in range(3):
    plt.subplot(3,1,i+1)
    longueur=imagettes_size["longueur"][imagettes_size["classe"]==liste_birds[i]]
    largeur=imagettes_size["largeur"][imagettes_size["classe"]==liste_birds[i]]
    
    rapport=longueur/largeur
    print(np.mean(rapport))
    plt.title(liste_birds[i])
    plt.hist(rapport)
    
    


for i in range(3):
    plt.subplot(3,1,i+1)
    longueur=imagettes_size["longueur"][imagettes_size["classe"]==liste_birds[i]]
    largeur=imagettes_size["largeur"][imagettes_size["classe"]==liste_birds[i]]
    
    rapport=longueur/largeur
    print(np.mean(rapport))
    plt.title(liste_birds[i])
    #plt.hist(longueur)
    plt.hist(largeur)