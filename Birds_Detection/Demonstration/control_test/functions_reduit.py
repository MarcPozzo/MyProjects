#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:29:41 2020

@author: marcpozzo
"""
import ast
import pandas as pd
import cv2


fichierClasses= "../Materiel/Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)

def to_reference_labels (df,class_colum,frame=frame):

    #flatten list in Labels_File
    cat=[]
    for i in range(len(frame["categories"]) ):
        cat.append( frame["categories"][i] )

    liste = [ast.literal_eval(item) for item in cat]

    # set nouvelle_classe to be the "unified" class name
    for j in range(len(frame["categories"])):
        #classesToReplace = frame["categories"][j].split(",")[0][2:-1]
        className = frame["categories"][j].split(",")[0][2:-1]
        #df["nouvelle_classe"]=df["classe"].replace(classesToReplace,className)
        df[class_colum]=df[class_colum].replace(liste[j],className)

    return df


def draw_rectangle(table,color,image_name):

    for i in range(len(table)):
        xmin=table["xmin"].iloc[i]
        ymin=table["ymin"].iloc[i]
        xmax=table["xmax"].iloc[i]
        ymax=table["ymax"].iloc[i]
        
        if color=="Blue":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
        if color=="Green":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 
        if color=="Red":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (0, 0,255), 2) 
        if color=="White":
            cv2.rectangle(image_name, (xmin,ymin), (xmax,ymax), (255, 255,255), 2)
        
    return image_name
            
            
