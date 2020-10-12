import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import math
import common
import config
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import common
tf.enable_eager_execution
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)



fold_to_keep=[       './DonneesPI/timeLapsePhotos_Pi1_4',
       './DonneesPI/timeLapsePhotos_Pi1_3',
       './DonneesPI/timeLapsePhotos_Pi1_2',
       './DonneesPI/timeLapsePhotos_Pi1_1',
       './DonneesPI/timeLapsePhotos_Pi1_0']




Mat_path="../../Materiels/"
neurone="training_jeux_difficile"
string=Mat_path+neurone

path_to_proj="../../"
imagettes=pd.read_csv(path_to_proj+"Materiels/"+"imagettes.csv")
imagettes=imagettes[imagettes["path"].isin(fold_to_keep)]
imagettes=common.to_reference_labels (imagettes,"classe")
index_train,index_test=common.split(imagettes)


#Choose  index_test or index_train
index=index_test
im_file=imagettes[imagettes["filename"].isin(index)].iloc[0:2]
images, labels, labels2=common.read_imagettes(im_file)
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)

dataset=tf.data.Dataset.from_tensor_slices((images, labels)).batch(config.batch_size)


Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)
checkpoint=tf.train.Checkpoint(model=Model)
checkpoint.restore(tf.train.latest_checkpoint(string))





start=time.time()
nb_images_to_catch=len(imagettes[imagettes["filename"].isin(index)])

seuil_liste=[0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
seuil_liste=[0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.12,0.14,0.16,0.18,0.2]
FP=[]
IC=[]
IP=[]

for seuil in seuil_liste:
        
    
    score,tp_nb,nr_rep,pres,box_caught=common.calcul_map(Model, dataset,labels2,seuil=seuil)
    
    #print("duree", end-start)
    
    print("XXXXXXXXXXXXXXXXXXXXX")
    print("Le seuil est: ", seuil)
    ic=round((pres/nb_images_to_catch)*100,2)
    IC.append(ic)
    print("pourcentage d'image capturé",ic)
    ip=round((tp_nb/nb_images_to_catch)*100,2)
    IP.append(ip)
    print("pourcentage d'image bien prédites",ip)
    print("nb rep",nr_rep)
    print("pres",pres)
    fp=nr_rep-pres
    FP.append(fp)
    print("le nombre de faux positifs pour une image analysée",fp/len(images))
    #print("pourcentage de réponse sup",)


end=time.time()

"""
Avec bruit 

pourcentage d'image capturé 32.09
pourcentage d'image bien prédites 28.57

Sans bruit 

pourcentage d'image capturé 43.74
pourcentage d'image bien prédites 39.34
le nombre de faux positifs pour une image analysée 0.479704797
le nombre de faux positifs pour une image analysée 0.6273062730627307


Pour le train 

temps
duree 36.58014178276062


Avec bruit

pourcentage d'image capturé 61.37
pourcentage d'image bien prédites 60.81
le nombre de faux positifs pour une image analysée 0.4305555555555556

Sans bruit

pourcentage d'image capturé 89.22
pourcentage d'image bien prédites 89.16
le nombre de faux positifs pour une image analysée 0.03425925925925926


"""
