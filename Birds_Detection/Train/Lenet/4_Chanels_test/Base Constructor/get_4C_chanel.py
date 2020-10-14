#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:30:46 2020

@author: marcpozzo
"""

import pandas as pd
import functions_4C as fn
import time
from sklearn.model_selection import train_test_split
import gc
from numpy import save


test_size=0.2
path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/"
type_of_diff="GBR"



liste_folders=['./DonneesPI/timeLapsePhotos_Pi1_0','./DonneesPI/timeLapsePhotos_Pi1_1','./DonneesPI/timeLapsePhotos_Pi1_2','./DonneesPI/timeLapsePhotos_Pi1_3','./DonneesPI/timeLapsePhotos_Pi1_4']

df=pd.read_csv(path+"images_labels.csv")
df=fn.open_table(df,liste_folders)

liste_name_test,filename_liste=list(fn.get_liste_name_test(df))
df["filename"]=list(fn.get_liste_name_test(df))[0]


base_train,base_test= train_test_split(df,stratify=df["classe"], test_size=test_size,random_state=42)
del df
gc.collect()


start=time.time()
X_train,Y_train,base_train_trans=fn.get_X_Y_bis(base_train,diff=type_of_diff)
end=time.time()

print(end-start)
save('imagettes4C_train.npy', X_train)

gc.collect()

