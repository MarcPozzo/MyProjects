import os
from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface_mark/test_Yolo/6_classes_loss/")
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

batch_size=16


imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon"]
imagettes=common.to_reference_labels (imagettes,"classe")
imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]
folder_to_keep= ['./DonneesPI/timeLapsePhotos_Pi1_4',
       './DonneesPI/timeLapsePhotos_Pi1_3',
       './DonneesPI/timeLapsePhotos_Pi1_2',
       './DonneesPI/timeLapsePhotos_Pi1_1',
       './DonneesPI/timeLapsePhotos_Pi1_0']   
imagettes=imagettes[imagettes["path"].isin(folder_to_keep)]
imagettes['freq'] = imagettes.groupby('filename')['filename'].transform('count')
imagettes.groupby('classe')['classe'].transform('count')







imagettes["cat_maj"]=0
liste_name_test=imagettes["filename"].unique()
for name_test in liste_name_test:
    list_cat=list(imagettes["classe"][imagettes["filename"]==name_test].unique())
    cat_maj=list_cat[0]
    imagettes["cat_maj"][imagettes["filename"]==name_test]=cat_maj

dataframe =imagettes.sort_values('filename').drop_duplicates(subset=['filename'])

fn_train,fn_test=train_test_split(dataframe["filename"],stratify=dataframe[['path', 'classe']],random_state=42,test_size=0.2)

imagettes_train=imagettes[imagettes["filename"].isin(fn_test)]
images_train, labels_train, labels2_train=common.read_imagettes(imagettes_train)



images_train=np.array(images_train, dtype=np.float32)/255
labels_train=np.array(labels_train, dtype=np.float32)


print("Nbr images:", len(images_train))

#train_ds=tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)
train_ds=tf.data.Dataset.from_tensor_slices((images_train, labels_train)).batch(batch_size)

def my_loss(labels, preds,labels2):
    grid=tf.meshgrid(tf.range(config.cellule_x, dtype=tf.float32), tf.range(config.cellule_y, dtype=tf.float32))
    grid=tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    grid=tf.tile(grid, (1, 1, config.nbr_boxes, 1))
    
    preds_xy    =tf.math.sigmoid(preds[:, :, :, :, 0:2])+grid
    preds_wh    =preds[:, :, :, :, 2:4]
    preds_conf  =tf.math.sigmoid(preds[:, :, :, :, 4])
    preds_classe=tf.math.sigmoid(preds[:, :, :, :, 5:])

    preds_wh_half=preds_wh/2
    preds_xymin=preds_xy-preds_wh_half
    preds_xymax=preds_xy+preds_wh_half
    preds_areas=preds_wh[:, :, :, :, 0]*preds_wh[:, :, :, :, 1]

    l2_xy_min=labels2[:, :, 0:2]
    l2_xy_max=labels2[:, :, 2:4]
    l2_area  =labels2[:, :, 4]
    
    preds_xymin=tf.expand_dims(preds_xymin, 4)
    preds_xymax=tf.expand_dims(preds_xymax, 4)
    preds_areas=tf.expand_dims(preds_areas, 4)

    labels_xy    =labels[:, :, :, :, 0:2]
    labels_wh    =tf.math.log(labels[:, :, :, :, 2:4]/config.anchors)
    labels_wh=tf.where(tf.math.is_inf(labels_wh), tf.zeros_like(labels_wh), labels_wh)
    
    conf_mask_obj=labels[:, :, :, :, 4]
    labels_classe=labels[:, :, :, :, 5:]
    
    conf_mask_noobj=[]
    for i in range(len(preds)):
        xy_min=tf.maximum(preds_xymin[i], l2_xy_min[i])
        xy_max=tf.minimum(preds_xymax[i], l2_xy_max[i])
        intersect_wh=tf.maximum(xy_max-xy_min, 0.)
        intersect_areas=intersect_wh[..., 0]*intersect_wh[..., 1]
        union_areas=preds_areas[i]+l2_area[i]-intersect_areas
        ious=tf.truediv(intersect_areas, union_areas)
        best_ious=tf.reduce_max(ious, axis=3)
        conf_mask_noobj.append(tf.cast(best_ious<config.seuil_iou_loss, tf.float32)*(1-conf_mask_obj[i]))
    conf_mask_noobj=tf.stack(conf_mask_noobj)

    preds_x=preds_xy[..., 0]
    preds_y=preds_xy[..., 1]
    preds_w=preds_wh[..., 0]
    preds_h=preds_wh[..., 1]
    labels_x=labels_xy[..., 0]
    labels_y=labels_xy[..., 1]
    labels_w=labels_wh[..., 0]
    labels_h=labels_wh[..., 1]

    loss_xy=tf.reduce_sum(conf_mask_obj*(tf.math.square(preds_x-labels_x)+tf.math.square(preds_y-labels_y)), axis=(1, 2, 3))
    loss_wh=tf.reduce_sum(conf_mask_obj*(tf.math.square(preds_w-labels_w)+tf.math.square(preds_h-labels_h)), axis=(1, 2, 3))

    loss_conf_obj=tf.reduce_sum(conf_mask_obj*tf.math.square(preds_conf-conf_mask_obj), axis=(1, 2, 3))
    loss_conf_noobj=tf.reduce_sum(conf_mask_noobj*tf.math.square(preds_conf-conf_mask_obj), axis=(1, 2, 3))

    loss_classe=tf.reduce_sum(tf.math.square(preds_classe-labels_classe), axis=4)
    loss_classe=tf.reduce_sum(conf_mask_obj*loss_classe, axis=(1, 2, 3))
    
    loss=config.lambda_coord*loss_xy+config.lambda_coord*loss_wh+loss_conf_obj+config.lambda_noobj*loss_conf_noobj+loss_classe
    return loss

model=model.model(config.nbr_classes, config.nbr_boxes, config.cellule_y, config.cellule_x)

string="./training/"
@tf.function
def train_step(images,labels,labels2):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=my_loss(labels, predictions,labels2)
  gradients=tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)

#def train(train_ds, nbr_entrainement,string,labels,labels2):
def train(train_ds, nbr_entrainement,string,labels2):
    for entrainement in range(nbr_entrainement):
        start=time.time()
        for images, labels in train_ds:
            train_step(images, labels,labels2)
        message='Entrainement {:04d}: loss: {:6.4f}, temps: {:7.4f}'
        print(message.format(entrainement+1,
                             train_loss.result(),
                             time.time()-start))
        if not entrainement%20:
            checkpoint.save(file_prefix=string)
    
optimizer=tf.keras.optimizers.Adam(learning_rate=1E-4)
checkpoint=tf.train.Checkpoint(model=model)
train_loss=tf.keras.metrics.Mean()

checkpoint=tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(string))


#train(train_ds, 400)
#train_test_split(train_ds)
start=time.time()
train(train_ds, 1,string,labels2_train)
end=time.time()
print((end-start)/60)
checkpoint.save(file_prefix=string)






