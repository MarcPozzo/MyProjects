import os

import tensorflow as tf
import sys
import time
import cv2
import numpy as np
import common
import config
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import common
import common_2

batch_size=16




tf_version=tf.__version__
if tf_version[0]=="1":
    tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None)
elif tf_version[0]=="2":
    print("Warning your are in tensorflow 2, it the programm doesn't work please try with tensorflow 1")
    tf.enable_eager_execution

path_model_saved="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/Yolo_models/"


imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
imagettes=common.to_reference_labels (imagettes,"classe")
index_train,index_test=common.split(imagettes)
imagettes_train=imagettes[imagettes["filename"].isin(index_train[:5])]
imagettes_train["filename"]






images_2, labels_2, labels2_2=common_2.read_imagettes(imagettes_train,nbr=5,coef=1.3,flip=True)
images_2=np.array(images_2, dtype=np.float32)/255
labels_2=np.array(labels_2, dtype=np.float32)
train_ds_2=tf.data.Dataset.from_tensor_slices((images_2, labels_2)).batch(batch_size)



Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)

# note: plein de modèles entrainés qui sont mis dans différents sous-répertoires
# un sous-répertoire correspond à un modèle pour Yolo
string=path_model_saved+"generateur_avec_flip_1000/"

    
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
checkpoint=tf.train.Checkpoint(model=Model)
train_loss=tf.keras.metrics.Mean()

checkpoint=tf.train.Checkpoint(model=Model)
checkpoint.restore(tf.train.latest_checkpoint(string))


#train(train_ds, 400)
#train_test_split(train_ds)
start=time.time()
nbr_entrainement=1
LOSS=common_2.train(train_ds_2, nbr_entrainement,string,labels2_2,optimizer,Model,train_loss,checkpoint)
end=time.time()
print((end-start)/60)
checkpoint.save(file_prefix=string)
#shutil.rmtree('training_bruit/')





