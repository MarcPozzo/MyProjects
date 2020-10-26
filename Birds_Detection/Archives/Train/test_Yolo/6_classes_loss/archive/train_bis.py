import os
from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/test_Yolo/6_classes_loss/")
import tensorflow as tf
import sys
import time
import cv2
import numpy as np
#import common_2 as common
import common_2 as common
import config
import model
import pandas as pd
from sklearn.model_selection import train_test_split

batch_size=16
nbr_entrainement=1

imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes=common.to_reference_labels (imagettes,"classe")
index_train,index_test=common.split(imagettes)
imagettes_train=imagettes[imagettes["filename"].isin(index_train)]


images, labels, labels2=common.read_imagettes(imagettes_train,nbr=1)
images=np.array(images, dtype=np.float32)/255
labels=np.array(labels, dtype=np.float32)
print("Nbr images:", len(images))
train_ds=tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)


Model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)

string="/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/Models/Yolo_models/training/"


    
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
checkpoint=tf.train.Checkpoint(model=Model)
train_loss=tf.keras.metrics.Mean()
checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(string))







start=time.time()
end=time.time()
print((end-start)/60)
checkpoint.save(file_prefix=string)
#common.train(train_ds, nbr_entrainement,string,labels2)


common.train(train_ds, nbr_entrainement,string,labels2_train,optimizer,model,train_loss,checkpoint)





optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
checkpoint=tf.train.Checkpoint(model=model)
train_loss=tf.keras.metrics.Mean()

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(string))


#train(train_ds, 400)
#train_test_split(train_ds)
start=time.time()
train(train_ds, 600,string,labels2_train)
end=time.time()
print((end-start)/60)
checkpoint.save(file_prefix=string)