path_to_proj='/Users/marcpozzo/Documents/Projet_Git/Projet_Git/Birds_Detection/'
path_Yolo2="Train/test_Yolo/6_classes_loss"

path=path_to_proj+path_Yolo2

path=path_to_proj+path_Yolo2
from os import chdir
chdir(path)
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import random
import cv2
import numpy as np
import math
import config
import pandas as pd
from pandas.core.common import flatten
from sklearn.model_selection import train_test_split
import ast
import time
#
fichierClasses= path_to_proj+"Materiel/Table_Labels_to_Class.csv" # overwritten by --classes myFile
frame=pd.read_csv(fichierClasses,index_col=False)

def to_reference_labels (df,class_colum,frame=frame):
    
    
    #Select the Pi images
    folder_to_keep= ['./DonneesPI/timeLapsePhotos_Pi1_4',
       './DonneesPI/timeLapsePhotos_Pi1_3',
       './DonneesPI/timeLapsePhotos_Pi1_2',
       './DonneesPI/timeLapsePhotos_Pi1_1',
       './DonneesPI/timeLapsePhotos_Pi1_0']   
    df=df[df["path"].isin(folder_to_keep)]

    ## Transform labels in workable labels
    #flatten list in Labels_File
    cat=[]
    for i in range(len(frame["categories"]) ):
        cat.append( frame["categories"][i] )

    liste = [ast.literal_eval(item) for item in cat]

    print("check point")
    # set nouvelle_classe to be the "unified" class name
    for j in range(len(frame["categories"])):
        #classesToReplace = frame["categories"][j].split(",")[0][2:-1]
        className = frame["categories"][j].split(",")[0][2:-1]
        #df["nouvelle_classe"]=df["classe"].replace(classesToReplace,className)
        df[class_colum]=df[class_colum].replace(liste[j],className)
        
    print("check point2")
    #Select only categories with enough values
    liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon"]
    df=df[df["classe"].isin(liste_to_keep)]
    print("check point3")
    return df

def sigmoid(x):
  x=np.clip(x, -50, 50)
  return 1/(1+np.exp(-x))

def softmax(x):
  e=np.exp(x)
  e_sum=np.sum(e)
  return e/e_sum

def prepare_image(image, labels, grille=True):
  img=image.copy()
  
  if grille is True:
    for x in range(config.r_x, config.largeur+config.r_x, config.r_x):
      for y in range(config.r_y, config.hauteur+config.r_y, config.r_y):
        cv2.line(img, (0, y), (x, y), (0, 0, 0), 1)
        cv2.line(img, (x, 0), (x, y), (0, 0, 0), 1)

  for y in range(config.cellule_y):
    for x in range(config.cellule_x):
      for box in range(config.nbr_boxes):
        if labels[y, x, box, 4]:
          ids=np.argmax(labels[y, x, box, 5:])
          
          x_center=int(round(labels[y, x, box, 0]*config.r_x))
          y_center=int(round(labels[y, x, box, 1]*config.r_y))
          print(x_center,y_center)
          w_2=int(round(labels[y, x, box, 2]*config.r_x/2))
          h_2=int(round(labels[y, x, box, 3]*config.r_y/2))
          x_min=x_center-w_2
          y_min=y_center-h_2
          x_max=x_center+w_2
          y_max=y_center+h_2
          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), list(config.dict.values())[ids], 1)
          cv2.circle(img, (x_center, y_center), 1, list(config.dict.values())[ids], 2)
            
  return img


def prepare_image_debug(image, labels, grille=True):
  img=image.copy()
  
  if grille is True:
    for x in range(config.r_x, config.largeur+config.r_x, config.r_x):
      for y in range(config.r_y, config.hauteur+config.r_y, config.r_y):
        cv2.line(img, (0, y), (x, y), (0, 0, 0), 1)
        cv2.line(img, (x, 0), (x, y), (0, 0, 0), 1)

  for y in range(config.cellule_y):
    for x in range(config.cellule_x):
      for box in range(config.nbr_boxes):
        if labels[y, x, box, 4]:
          ids=np.argmax(labels[y, x, box, 5:])
          
          x_center=int(round(labels[y, x, box, 0]*config.r_x))
          y_center=int(round(labels[y, x, box, 1]*config.r_y))
          print(x_center,y_center)
          w_2=int(round(labels[y, x, box, 2]*config.r_x/2))
          h_2=int(round(labels[y, x, box, 3]*config.r_y/2))
          x_min=x_center-w_2
          y_min=y_center-h_2
          x_max=x_center+w_2
          y_max=y_center+h_2
          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), list(config.dict.values())[ids], 1)
          cv2.circle(img, (x_center, y_center), 1, list(config.dict.values())[ids], 2)
            
  return img

def bruit(image):
  h, w, c=image.shape
  #n=np.random.randn(h, w, c)*random.randint(5, 30)
  n=np.random.randn(h, w, c)*random.randint(1, 10)
  return np.clip(image+n, 0, 255).astype(np.uint8)

def gamma(image, alpha=1.0, beta=0.0):
  return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

def intersection_over_union(boxA, boxB):
  xA=np.maximum(boxA[0], boxB[0])
  yA=np.maximum(boxA[1], boxB[1])
  xB=np.minimum(boxA[2], boxB[2])
  yB=np.minimum(boxA[3], boxB[3])
  interArea=np.maximum(0, xB-xA)*np.maximum(0, yB-yA)
  boxAArea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
  boxBArea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
  return interArea/(boxAArea+boxBArea-interArea)





imagettes=pd.read_csv(path_to_proj+"Materiel/"+"imagettes.csv")

   

def read_imagettes(imagettes):
  images=[]
  labels=[]
  labels2=[]
  
  imagettes_copy=imagettes.copy()

  liste_name_test=list(imagettes["filename"].unique())

  for name_test in liste_name_test:
      image, label, label2=prepare_labels_marc(name_test, imagettes_copy)
     
         
      if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
 
    
  images=np.array(images)
  labels=np.array(labels)
  labels2=np.array(labels2)
  return images, labels, labels2  


 
#  
"""
def read_imagettes(imagettes):
  images=[]
  labels=[]
  labels2=[]
  
  imagettes_copy=imagettes.copy()
  print("tail tab_init",len( imagettes_copy[ imagettes_copy["filename"]=='image_2019-04-22_19-11-23.jpg']))
  liste_name_test=list(imagettes["filename"].unique())

  for name_test in liste_name_test:
      image, label, label2=prepare_labels_marc(name_test, imagettes_copy)
      if name_test=='image_2019-04-22_19-11-23.jpg':
          print("tail tab",len( imagettes_copy[ imagettes_copy["filename"]==name_test]))
       
          print("objet",label2[1])
         
      if image is not None:
          images.append(image)
          labels.append(label)
          labels2.append(label2)
 
    
  images=np.array(images)
  labels=np.array(labels)
  labels2=np.array(labels2)
  return images, labels, labels2    
"""    

def prepare_labels_marc(name_test,imagettes):



    imagettes_copy=imagettes.copy()
    One_image=imagettes_copy[imagettes_copy["filename"]==name_test]
    #if name_test=='image_2019-04-22_19-11-23.jpg':
    #    print("taille table",len(One_image))
    
    #path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
    path_base="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises"
    path_folder=One_image["path"].iloc[0][1:]+"/"
    path=path_base+path_folder
    #big_image_path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-55-20.jpg"
    big_image_path=path+name_test
    big_image=cv2.imread(big_image_path)
      
    coeff=1
    image_r=cv2.resize(big_image, (int(round(coeff*config.largeur)), int(round(coeff*config.hauteur))))
    #image_r=gamma(image_r, random.uniform(0.7, 1.3), np.random.randint(60)-30)
    #image_r=bruit(image_r)

    shift_x=0
    shift_y=0
    
    ratio_x=coeff*config.largeur/big_image.shape[1]
    ratio_y=coeff*config.hauteur/big_image.shape[0]


    label =np.zeros((config.cellule_y, config.cellule_x, config.nbr_boxes, 5+config.nbr_classes), dtype=np.float32)
    label2=np.zeros((config.max_objet, 7), dtype=np.float32)
    

    nbr_objet=0
    for i in range(len(One_image)):
        One_imagette=One_image.iloc[i]
        classe=One_imagette["classe"]
        id_class=config.dict2.index(classe)
        (x_min,x_max,y_min,y_max)=One_imagette[["xmin","xmax","ymin","ymax"]]
        x_min=int((x_min*ratio_x))
        x_max=int((x_max*ratio_x))
        y_min=int((y_min*ratio_y))
        y_max=int((y_max*ratio_y))

            
        x_min=(x_min-shift_x)/config.r_x
        y_min=(y_min-shift_y)/config.r_y
        x_max=(x_max-shift_x)/config.r_x
        y_max=(y_max-shift_y)/config.r_y

        area=(x_max-x_min)*(y_max-y_min)
        label2[nbr_objet]=[x_min, y_min, x_max, y_max, area, 1, id_class]
        #if name_test=='image_2019-04-22_19-11-23.jpg':
        #    print("prem el lab2",label2[1])
        #    print(i)
        #nbr_objet+=1
        x_centre=x_min+(x_max-x_min)/2
        y_centre=y_min+(y_max-y_min)/2
        x_cell=int((x_centre))
        y_cell=int((y_centre))
        
             
        largeur=x_max-x_min
        hauteur=y_max-y_min
        #x_centre=int(x_min+(x_max-x_min)/2)
        #y_centre=int(y_min+(y_max-y_min)/2)
        #x_cell=int(x_centre)
        #y_cell=int(y_centre)

        a_x_min=x_centre-config.anchors[:, 0]/2
        a_y_min=y_centre-config.anchors[:, 1]/2
        a_x_max=x_centre+config.anchors[:, 0]/2
        a_y_max=y_centre+config.anchors[:, 1]/2

        id_a=0
        best_iou=0
        for i in range(len(config.anchors)):
          iou=intersection_over_union([x_min, y_min, x_max, y_max], [a_x_min[i], a_y_min[i], a_x_max[i], a_y_max[i]])
          if iou>best_iou:
            best_iou=iou
            id_a=i

        label[y_cell, x_cell, id_a, 0]=x_centre
        label[y_cell, x_cell, id_a, 1]=y_centre
        label[y_cell, x_cell, id_a, 2]=largeur
        label[y_cell, x_cell, id_a, 3]=hauteur
        label[y_cell, x_cell, id_a, 4]=1.
        label[y_cell, x_cell, id_a, 5+id_class]=1.
        
        nbr_objet=nbr_objet+1
        if nbr_objet==config.max_objet:
          print("Nbr objet max atteind !!!!!")
          break
    #if name_test=='image_2019-04-22_19-11-23.jpg':
    #    print("c etait la bonne",label2[1])
    return image_r,label,label2



#Inquire the prior categories
def select_one_category(list_cat):
    if len(list_cat)==1:
        category=list_cat[0]
        
    if len(list_cat)>1.1:
        if "faisan" in list_cat:
            category="faisan"
        elif "pigeon" in list_cat:
            category="pigeon"  
        elif "corneille" in list_cat:
            category="corneille"
        elif "lapin" in list_cat:
            category="lapin"             
        elif "chevreuil" in list_cat:
            category="chevreuil" 
    return category


#Split DataCategories with adaptapted stratify
def split(imagettes):
    
    liste_name_test=list(imagettes["filename"].unique())
    #get dic name_test less birds vs represented categories
    dic_name_test={}
    for name_test in liste_name_test:
        liste_animals=list(imagettes["classe"][imagettes["filename"]==name_test].values)
        dic_name_test[name_test]=select_one_category(liste_animals)

    #fill df with dic
    imagettes["cat_maj"]=0
    for ind in imagettes.index:
        imagettes["cat_maj"].loc[ind]=dic_name_test[imagettes["filename"].loc[ind]]
    
    #Split the DataSet
    dataframe =imagettes.sort_values('filename').drop_duplicates(subset=['filename'])
    fn_train,fn_test=train_test_split(dataframe["filename"],stratify=dataframe[['path', 'cat_maj']],random_state=42,test_size=0.2)
    
    return fn_train,fn_test


"""


def prepare_labels_marc(name_test,imagettes):



 
    One_image=imagettes[imagettes["filename"]==name_test]
    #path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
    path_base="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    path_folder=One_image["path"].iloc[0][1:]+"/"
    path=path_base+path_folder
    #big_image_path="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-55-20.jpg"
    big_image_path=path+name_test
    big_image=cv2.imread(big_image_path)
      
    coeff=1
    image_r=cv2.resize(big_image, (int(round(coeff*config.largeur)), int(round(coeff*config.hauteur))))

    shift_x=0
    shift_y=0
    
    ratio_x=coeff*config.largeur/big_image.shape[1]
    ratio_y=coeff*config.hauteur/big_image.shape[0]


    label =np.zeros((config.cellule_y, config.cellule_x, config.nbr_boxes, 5+config.nbr_classes), dtype=np.float32)
    #Le 7 correspond à 7 objets et non 7 classes, la classe est le dernier de ces objets
    label2=np.zeros((config.max_objet, 7), dtype=np.float32)
    
    nbr_objet=0
    for i in range(len(One_image)):
        One_imagette=One_image.iloc[i]
        classe=One_imagette["classe"]
        id_class=config.dict2.index(classe)
        (x_min,x_max,y_min,y_max)=One_imagette[["xmin","xmax","ymin","ymax"]]
        x_min=int(round(x_min*ratio_x))
        x_max=int(round(x_max*ratio_x))
        y_min=int(round(y_min*ratio_y))
        y_max=int(round(y_max*ratio_y))

            
        x_min=(x_min-shift_x)/config.r_x
        y_min=(y_min-shift_y)/config.r_y
        x_max=(x_max-shift_x)/config.r_x
        y_max=(y_max-shift_y)/config.r_y

        area=(x_max-x_min)*(y_max-y_min)
        label2[nbr_objet]=[x_min, y_min, x_max, y_max, area, 1, id_class]
        
        
        x_centre=x_min+(x_max-x_min)/2
        y_centre=y_min+(y_max-y_min)/2
        x_cell=int((x_centre))
        y_cell=int((y_centre))
        
        if x_cell==16:
            x_cell=15
        if y_cell==12:
            y_cell=11            
        largeur=x_max-x_min
        hauteur=y_max-y_min
        #x_centre=int(x_min+(x_max-x_min)/2)
        #y_centre=int(y_min+(y_max-y_min)/2)
        #x_cell=int(x_centre)
        #y_cell=int(y_centre)

        a_x_min=x_centre-config.anchors[:, 0]/2
        a_y_min=y_centre-config.anchors[:, 1]/2
        a_x_max=x_centre+config.anchors[:, 0]/2
        a_y_max=y_centre+config.anchors[:, 1]/2

        id_a=0
        best_iou=0
        for i in range(len(config.anchors)):
          iou=intersection_over_union([x_min, y_min, x_max, y_max], [a_x_min[i], a_y_min[i], a_x_max[i], a_y_max[i]])
          if iou>best_iou:
            best_iou=iou
            id_a=i

        label[y_cell, x_cell, id_a, 0]=x_centre
        label[y_cell, x_cell, id_a, 1]=y_centre
        label[y_cell, x_cell, id_a, 2]=largeur
        label[y_cell, x_cell, id_a, 3]=hauteur
        label[y_cell, x_cell, id_a, 4]=1.
        label[y_cell, x_cell, id_a, 5+id_class]=1.
        
        nbr_objet=nbr_objet+1
        if nbr_objet==config.max_objet:
          print("Nbr objet max atteind !!!!!")
          break

    return image_r,label,label2
"""



def calcul_map(Model, dataset,labels2, beta=1., seuil=0.1):
  tp_nb=0
  pres=0
  box_caught=0
  nr_rep=0
  grid=np.meshgrid(np.arange(config.cellule_x, dtype=np.float32), np.arange(config.cellule_y, dtype=np.float32))
  grid=np.expand_dims(np.stack(grid, axis=-1), axis=2)
  grid=np.tile(grid, (1, 1, 1, config.nbr_boxes, 1))

  index_labels2=0
  labels2_=labels2*[config.r_x, config.r_y, config.r_x, config.r_y, 1, 1, 1]
  score=[]
  tab_nbr_reponse=[]
  tab_tp=[]
  tab_true_boxes=[]
  
  for images, labels in dataset:
    predictions=np.array(Model(images))

    pred_conf=sigmoid(predictions[:, :, :, :, 4])
    pred_classes=softmax(predictions[:, :, :, :, 5:])
    pred_ids=np.argmax(pred_classes, axis=-1)
    
    x_center=((grid[:, :, :, :, 0]+sigmoid(predictions[:, :, :, :, 0]))*config.r_x)
    y_center=((grid[:, :, :, :, 1]+sigmoid(predictions[:, :, :, :, 1]))*config.r_y)
    w=(np.exp(predictions[:, :, :, :, 2])*config.anchors[:, 0]*config.r_x)
    h=(np.exp(predictions[:, :, :, :, 3])*config.anchors[:, 1]*config.r_y)

    x_min=x_center-w/2
    y_min=y_center-h/2
    x_max=x_center+w/2
    y_max=y_center+h/2

    tab_boxes=np.stack([y_min, x_min, y_max, x_max], axis=-1).astype(np.float32)
    tab_boxes=tab_boxes.reshape(-1, config.cellule_y*config.cellule_x*config.nbr_boxes, 4)
    pred_conf=pred_conf.reshape(-1, config.cellule_y*config.cellule_x*config.nbr_boxes)
    pred_ids=pred_ids.reshape(-1, config.cellule_y*config.cellule_x*config.nbr_boxes)

    for p in range(len(predictions)):
      nbr_reponse=np.zeros(config.nbr_classes)
      tp=np.zeros(config.nbr_classes)
      nbr_true_boxes=np.zeros(config.nbr_classes)
      tab_index=tf.image.non_max_suppression(tab_boxes[p], pred_conf[p], 100)      
      for id in tab_index:
        if pred_conf[p, id]>seuil:
           
          nbr_reponse[pred_ids[p, id]]+=1
          nr_rep+=1
          for box in labels2_[index_labels2]:
            #Aire vide
            if not box[5]:
              break
            b1=[tab_boxes[p, id, 1], tab_boxes[p, id, 0], tab_boxes[p, id, 3], tab_boxes[p, id, 2]]
            iou=intersection_over_union(b1, box)
            #La condition est ici donc c'est surtout dans un premier temps le deuxième point qu'on veut vérifier on va donc mettre un petit seuil
            if iou>seuil and box[6]==pred_ids[p, id]:
              tp[pred_ids[p, id]]+=1
              tp_nb+=1
            if iou>seuil:
                pres+=1
      for box in labels2[index_labels2]:
        #Aire vide  
        if not box[5]:
          break
        #box[6] la classe prédite
        nbr_true_boxes[int(box[6])]+=1
        box_caught+=1

      tab_nbr_reponse.append(nbr_reponse)
      tab_tp.append(tp)
      tab_true_boxes.append(nbr_true_boxes)
      
      index_labels2=index_labels2+1

  #tab_nbr_reponse c'est celui pour les tp.
  tab_nbr_reponse=np.array(tab_nbr_reponse)
  #Hypothès tab_true dit si on prédit au bon endroit
  #tab_tp si on prédit on bonne endroit le bon label
  tab_tp=np.array(tab_tp)
  tab_true_boxes=np.array(tab_true_boxes)

  ########################
  precision_globule_rouge=tab_tp[:, 1]/(tab_nbr_reponse[:, 1]+1E-7)
  precision_trophozoite=tab_tp[:, 4]/(tab_nbr_reponse[:, 4]+1E-7)

  rappel_globule_rouge=tab_tp[:, 1]/(tab_true_boxes[:, 1]+1E-7)
  rappel_trophozoite=tab_tp[:, 4]/(tab_true_boxes[:, 4]+1E-7)
    
 # print("F1 score globule rouge", np.mean(2*precision_globule_rouge*rappel_globule_rouge/(precision_globule_rouge+rappel_globule_rouge+1E-7)))
  #print("F1 score trophozoite", np.mean(2*precision_trophozoite*rappel_trophozoite/(precision_trophozoite+rappel_trophozoite+1E-7)))
  
  precision=(precision_globule_rouge+precision_trophozoite)/2
  rappel=(rappel_globule_rouge+rappel_trophozoite)/2

  score=np.mean((1+beta*beta)*precision*rappel/(beta*beta*precision+rappel+1E-7))
  #print("SCORE (globule rouge/trophozoite)", score)
  ########################

  precision=tab_tp/(tab_nbr_reponse+1E-7)
  rappel=tab_tp/(tab_true_boxes+1E-7)
  score=np.mean((1+beta*beta)*precision*rappel/(beta*beta*precision+rappel+1E-7))
  
  #nr_rep ne change pas 
  return score,tp_nb,nr_rep,pres,box_caught






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


#@tf.function
def train_step(images,labels,labels2,optimizer,model,train_loss):
  with tf.GradientTape() as tape:
    predictions=model(images)
    loss=my_loss(labels, predictions,labels2)
  gradients=tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)

#def train(train_ds, nbr_entrainement,string,labels,labels2):
def train(train_ds, nbr_entrainement,string,labels2,optimizer,model,train_loss,checkpoint):
    for entrainement in range(nbr_entrainement):
        start=time.time()
        for images, labels in train_ds:
            train_step(images, labels,labels2,optimizer,model,train_loss)
        message='Entrainement {:04d}: loss: {:6.4f}, temps: {:7.4f}'
        print(message.format(entrainement+1,
                             train_loss.result(),
                             time.time()-start))
        if not entrainement%20:
            checkpoint.save(file_prefix=string)