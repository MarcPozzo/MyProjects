#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:12:03 2020

@author: marcpozzo
"""

#Libraries
import functions_reduit as fn
import pandas as pd
import matplotlib.pyplot as plt
import cv2


#Picture you want watch annotations
name_test='image_2019-06-14_18-16-52.jpg'



#open animals tables
imagettes_path= "../Materiel/imagettes.csv" 
imagettes=pd.read_csv(imagettes_path,index_col=False)
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
imagettes=fn.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]   

#Read Picture you want to see
imagettes=imagettes[imagettes["filename"]==name_test]
path_folder=imagettes["path"].iloc[0][1:]
imageB=cv2.imread("../../.."+path_folder + "/"+name_test)


#imagettes_test=imagettes[imagettes["filename"]==name_test]

#Plot picture with annotations
imagettes_square=imagettes[['xmin', 'ymin', 'xmax','ymax']]
annoted_picture=fn.draw_rectangle(imagettes_square,"Blue",imageB)
plt.imshow(annoted_picture)

