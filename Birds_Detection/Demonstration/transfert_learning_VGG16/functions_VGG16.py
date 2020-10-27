#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:36:31 2020

@author: marcpozzo
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:36:31 2020
@author: marcpozzo
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf





def open_imagettes(Images,liste_name_test,data_path):


    imagettes_=[]
    Imagettes_copy=Images.copy()
    
    i=0
    for name_test in liste_name_test:
        i+=1
        One_image=Imagettes_copy[Imagettes_copy["filename"]==name_test]
        image_path=data_path+name_test
        image=cv2.imread(image_path)
        if image is None:
            print(name_test,i)
        
        for i in range(len(One_image)):
            imagette=One_image.iloc[i]
            xmin, ymin, xmax,ymax=imagette[['xmin', 'ymin', 'xmax','ymax']]
            imagette=image[ymin:ymax,xmin:xmax]
            imagette=tf.keras.preprocessing.image.img_to_array(imagette)

            imagette_r=cv2.resize(imagette, (224, 224))
            if imagette_r is not None:
                imagettes_.append(imagette_r)
                
    imagettes=np.array(imagettes_)       
    return imagettes






def get_features_to_df(images_df,model,liste_name_test,data_path,bird=True):
    
    #parameters
    NB_FEATURES=7 * 7 * 512 #Number of output in the fold
    list_birds=["corneille","faisan","pigeon","oiseau"] #classes to keep
    
    #Get output of VGG16
    images=open_imagettes(images_df,liste_name_test,data_path)
    features = model.predict(images)
    features_resize = features.reshape((features.shape[0],NB_FEATURES )) 
    features_name = ["f_"+str(i) for i in range(NB_FEATURES)]
   
    
    #Gather features and label in a dataframe
    if bird==True:
        label=1
        images_df=images_df[images_df["classe"].isin(list_birds)]
    if bird==False:
        label=0
        images_df=images_df[images_df["classe"].isin(list_birds)==False]
    column_names=features_name+["classe"]
    tableau_features = pd.DataFrame(columns = column_names) 
    for  vec in  features_resize:
        array=np.append(vec,label)
        tableau_features.loc[len(tableau_features)] = array
        
    return tableau_features



def get_tables(imagettes,model,liste_imagettes,data_path):
    tableau_birds_features=get_features_to_df(imagettes,model,liste_imagettes,data_path,bird=True)
    tableau_other_features=get_features_to_df(imagettes,model,liste_imagettes,data_path,bird=False)
    tableaux=[tableau_birds_features,tableau_other_features]
    tableau_features=pd.concat(tableaux)
    tableau_features=tableau_features.sample(frac=1).reset_index(drop=True)
    return tableau_features



