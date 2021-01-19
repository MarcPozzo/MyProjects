#This script is the support when it comes to test Lenet or VGG16
#The script is divided in several sections noted with ##:Prediction/Evaluation, Differences functions,Filters functions, (next below)
# Adjust extracted imagette, gather tiny images in a unique array,Hide unusefull part of the image

import cv2
import pandas as pd
import numpy as np
from numpy import load
from imutils import grab_contours
#from skimage.measure import compare_ssim
import math
#from keras.applications import VGG16
import joblib
import os
from os.path import basename, join



#Load models
Mat_path="../../Materiels/"
Mod_path=Mat_path+"Models/"
filtre_name=Mod_path+"RL_annotation_model"
#Model1 = joblib.load(Mod_path+"model.cpickle")
filtre_RL = joblib.load(filtre_name)
#coef_filtre=pd.read_csv(Mod_path+"coefs_filtre_RQ.csv")
#model = VGG16(weights="imagenet", include_top=False)

######################################################################################################################################################

##Prediction/Evaluation


#Predictions of birds with Lenet (with 3 and 4 chanels inputs) and evaluation the number of false , true positif ... . A mask can be set
def Evaluate_Lenet_prediction ( Images , name_test , name_ref  , CNNmodel , data_path ,dict_anotation_index_to_classe, ntarget_classes_,
                                   contrast , blockSize  , blurFact  ,seuil=210  ,
                       filtre_choice = "No_filtre" ,down_thresh = 25 ,
                      chanels = 3 , mask = False , 
                        thresh = 0.5,
                       diff_mod3C = "light" ,diff_mod4C = "HSV",precise=False):
                        
    

 
    #Opening images
    image_ref=data_path+name_ref
    image_test=data_path+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    Images_target=Images[Images["filename"]==name_test]
    Images_target=Images_target.drop('filename',axis=1)
    nb_oiseaux=len( Images_target)
    print("Birds in the picture: ",nb_oiseaux)
    
    #Set Mask    
    if mask==True:
        imageA=set_mask(imageA,folder="mask_path_to_fill",number_chanels=3)
        imageB=set_mask(imageA,folder="mask_path_to_fill",number_chanels=3)
        imageB=imageB.astype(np.uint8)
        imageA=imageA.astype(np.uint8)
       
    #differentiate between images and extract tiny images in areas of greatest difference
    tiny_images=diff_images(imageA,imageB,contrast=contrast,blockSize=blockSize,blurFact = blurFact,diff_mod3C=diff_mod3C,seuil=seuil)

    if  tiny_images :
        #reshape tiny_images in a shape adapted to Lenet and save localization in table_non_filtre
        batchImages_stack_reshape,table_non_filtre=batched_cnts(tiny_images,imageB)
        generate_square,batchImages_stack_reshape=filter_by_area(table_non_filtre,filtre_choice,batchImages_stack_reshape,down_thresh) #filters   
        #Add a 4th chanel (differention with the previous image)
        if chanels==4:
            Diff=diff_image4C(imageA,imageB,diff_mod4C=diff_mod4C)
            batchImages_stack_reshape=get_4C_all_batch(batchImages_stack_reshape,Diff,table_non_filtre)
    
        #We classify the generated tiny images according to the annotated coordinates. If this corresponds to an area with a bird it could be a false positive or a true positive
        Map_Target_Indexes=assign_timages_to_annotation(generate_square, Images_target,dict_anotation_index_to_classe) 
        #Ici par exemple on pourrait retourner les non targets, c'est à dire la liste de ceux qui sont dans la table en renvoyant leur numéro 
        
        
    else:
        batchImages_stack_reshape=[]
        generate_square=[]
    #Estimations established and classed
    (TP,FP)=(0,0)
    if batchImages_stack_reshape:
        estimates = CNNmodel.predict(np.array(batchImages_stack_reshape))
        predictions=list(estimates.argmax(axis=1))
        NB_ESTIMATES=len(estimates)
        indexes_targets_=list(Map_Target_Indexes["nv_index"])
        
        #Cout the TP
        for i in indexes_targets_:
            if precise==True:
                if predictions[i]== int(Map_Target_Indexes["nv_class"][Map_Target_Indexes["nv_index"]==i]):
                    TP+=1      
            elif precise==False:
                ESTIMATES_DEFAULTS_OBJECTS=sum(estimates[i][classe_defaut] for classe_defaut in ntarget_classes_)
                ESTIMATES_TARGETS=1-ESTIMATES_DEFAULTS_OBJECTS
                if ESTIMATES_TARGETS>thresh:
                    TP+=1
        
        #Cout the FP
        if precise==True:
            NB_DEFAULTS_LABELS_PREDICTIONS=0
            for d_indice in ntarget_classes_:
                NB_DEFAULTS_LABELS_PREDICTIONS+=predictions.count(d_indice)
            FP=NB_ESTIMATES   -NB_DEFAULTS_LABELS_PREDICTIONS  -len(Map_Target_Indexes) 
        elif precise==False:
            ntarget_indexes=set(range(NB_ESTIMATES))-set(indexes_targets_)
            pred_sum_ntargets_=[sum(estimates[i][classe_defaut] for classe_defaut in ntarget_classes_) for i in ntarget_indexes]
            FP=len([i for i in pred_sum_ntargets_ if i<1-thresh])
        
            
    return TP,FP





#Prediction
def Evaluate_extraction ( Images , name_test , name_ref ,data_path ,objects_targeted_, 
                      contrast = - 5 , blockSize = 53 , blurFact = 15 , 
                       diff_mod3C = "light",seuil = 210 ,mask = False ):
                        
    

    #Parameters
    #Dictionnary to convert string labels to num labels

    
    #Opening images
    image_ref=data_path+name_ref
    image_test=data_path+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    Objects_Images=Images[Images["classe"].isin(objects_targeted_)]
    Images_target=Objects_Images[Objects_Images["filename"]==name_test]
    Images_target=Images_target.drop('filename',axis=1)
    
    #Set Mask    
    if mask==True:
        imageA=set_mask(imageA,folder="mask_path_to_fill",number_chanels=3)
        imageB=set_mask(imageA,folder="mask_path_to_fill",number_chanels=3)
        imageB=imageB.astype(np.uint8)
        imageA=imageA.astype(np.uint8)
       
    #differentiate between images and extract tiny images in areas of greatest difference
    tiny_images=diff_images(imageA,imageB,contrast=contrast,blockSize=blockSize,blurFact = blurFact,diff_mod3C=diff_mod3C,seuil=seuil)

    if  tiny_images :
        #reshape tiny_images in a shape adapted to Lenet and save localization in table_non_filtre
        batchImages_stack_reshape,generate_square=batched_cnts(tiny_images,imageB)
        if len(generate_square)!=0:
            #We classify the generated tiny images according to the annotated coordinates. If this corresponds to an area with a bird it could be a false positive or a true positive
            an_caught,NB_OBJECTS_TO_CAUGHT,TINY_IMAGES_GENERATED=count_extractions(generate_square, Images_target) 

    return an_caught,NB_OBJECTS_TO_CAUGHT,TINY_IMAGES_GENERATED





######################################################################################################################################################
##Differences functions
    
#make difference between 3chanels images and return tiny images in areas of greatest difference
def diff_images(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25,diff_mod3C="light",seuil=210):
    
    if diff_mod3C=="light":
        imageA=imageA.astype(np.uint8)
        img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
        imageB=imageB.astype(np.uint8)
        img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    
        blurFact = blurFact
        absDiff2 = cv2.absdiff(img1, img2)
        diff = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        th2 = cv2.adaptiveThreshold(src=diff,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                                    thresholdType=cv2.THRESH_BINARY,blockSize=blockSize,C=contrast) #c=-30 pour la cam de chasse adaptation de C � histogram de la photo ?
    
        th2Blur=cv2.GaussianBlur(th2,(blurFact,blurFact),sigmaX=0)
        th2BlurTh = cv2.adaptiveThreshold(src=th2Blur,maxValue=255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                thresholdType=cv2.THRESH_BINARY,blockSize=blockSize,C=contrast) # adaptation de C � histogram de la photo ?
        threshS=th2BlurTh
    

        
    elif diff_mod3C=="ssim":
        
        a=cv2.GaussianBlur(imageA,(blurFact,blurFact),sigmaX=0)
        b=cv2.GaussianBlur(imageB,(blurFact,blurFact),sigmaX=0)
        
        (score, diff) = compare_ssim(a, b, full=True,multichannel=True)
        diff = (diff * 255).astype("uint8")
        blur = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite("ablur.jpg",blur)
        thresh = cv2.threshold(src=blur, thresh=seuil,maxval=255,type=cv2.THRESH_BINARY_INV)[1]
        
        threshS = cv2.dilate(thresh,(3,3))
        threshS = cv2.erode(threshS,(3,3),iterations=1)
        

    # defines corresponding regions of change
    cnts = cv2.findContours(threshS.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)    
    return cnts




#make difference between 4chanels images and return tiny images in areas of greatest difference
def diff_image4C(imageA,imageB,diff_mod4C="HSV"):
    if diff_mod4C=="HSV":
        img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
        img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
        absDiff2 = cv2.absdiff(img1, img2)
        diff_Gray = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        
    elif diff_mod4C=="BGR":
        absDiff2 = cv2.absdiff(imageA, imageB)
        diff_Gray = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        
    else:
        print("erreur de saisie de la methode")
        
    return diff_Gray


######################################################################################################################################################
##Filters functions

#return  only squares enought small to be birds
def filter_by_area(table_non_filtre,filtre_choice,batchImages_stack_reshape,down_thresh):

    #Drop the biggest squares
    small_squares=table_non_filtre.copy()
    xmin,xmax,ymin,ymax=get_table_coord(small_squares)
    small_squares["area"]=(xmax-xmin)*(ymax-ymin)
    small_squares=small_squares[small_squares["area"]<25000]
    small_squares=small_squares[small_squares["area"]>down_thresh]
    small_squares.drop(["area"],axis=1,inplace=True)
    
    #reindex if squares droped
    if len(small_squares)<len(table_non_filtre):
        batchImages_stack_reshape=[batchImages_stack_reshape[i] for i in small_squares.index ]
        #small_squares.reset_index(inplace=True)
    
    #Apply quantile filtre
    elif filtre_choice=="quantile_filtre":
        table_quantile,index_possible_animals=filtre_quantile(small_squares)
        small_squares=table_quantile
        batchImages_stack_reshape_quantile=[batchImages_stack_reshape[i] for i in index_possible_animals ]
        batchImages_stack_reshape=np.array(batchImages_stack_reshape_quantile)
        
    elif filtre_choice=="RL_filtre":
        small_squares=table_filtre_RL

    #Pas sûr à 100% qu'il faille le mettre ici
    small_squares.reset_index(inplace=True)
    return small_squares,batchImages_stack_reshape







######################################################################################################################################################
##Adjust extracted tiny images

#Recenters tiny images
def RecenterImage(subI,o):
    
    h,l,r=subI.shape #on sait que r=3 (3 channels)
    
    # add to the dimension the dimensions of the cuts (due to image borders)
    h = h + o.ymincut + o.ymaxcut
    l = l + o.xmincut + o.xmaxcut

    t= np.full((h, l, r), fill_value=int(round(subI.mean())),dtype=np.uint8) # black image the size of the final thing

    t[o.ymincut:(h-o.ymaxcut),o.xmincut:(l-o.xmaxcut)] = subI
    
    return t




def GetSquareSubset(img,h,verbose=True, xml = False):
    
   
    # determine le plus grand cete du carre
    d = max(h.ymax-h.ymin,h.xmax-h.xmin)
      
    # determine le centre du carre
    xcent = int(round((h.xmax-h.xmin)/2)) + h.xmin
    ycent = int(round((h.ymax-h.ymin)/2)) + h.ymin
    
    # new corners
    hd = int(math.ceil(d/2)) # half distance
    o = pd.Series(dtype='float64')
    o.xmin = xcent- hd
    o.xmax = xcent+ hd
    o.ymin = ycent- hd
    o.ymax = ycent+ hd
    """   
    print(o.xmin,o.xmax)
    print(o.ymin,o.ymax)
    """
    # check we are not further than the image borders
    # get the min/max accounting for image border + n cutted pixels: o.(x|y)(min|max)cut
    o.xmin,o.xmincut = accountForBorder(o.xmin,img.shape[1])
    o.xmax,o.xmaxcut = accountForBorder(o.xmax,img.shape[1])
    o.ymin,o.ymincut = accountForBorder(o.ymin,img.shape[0])
    o.ymax,o.ymaxcut = accountForBorder(o.ymax,img.shape[0])
    """
    print(o.xmin,o.xmax)
    print(o.ymin,o.ymax)
    """
    if(verbose):
        # dessine le carre
        cv2.rectangle(img, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #ecriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
        # add squares/numbers
    
    if(xml == True):
        # couper l'image
        subI = img[o.ymin:o.ymax,o.xmin:o.xmax]
    else:
        # couper l'image
        # subI = imageB[o.ymin:o.ymax,o.xmin:o.xmax]
        subI = img[o.ymin:o.ymax,o.xmin:o.xmax]
   
    
    return subI, o, d, img


 
    
  
# Verifie les bords de l'image pour la découpe    
def accountForBorder(val,maxval):

    if(val<0):
        cut = 0-val
        val = 0
    elif(val>maxval):
        cut = val - maxval
        val = maxval    
    else:
        cut = 0
    return val,cut


#area depending of coordonate
def area_square(x_min,x_max,y_min,y_max):
    

    
    profondeur=y_max-y_min
    largeur=x_max-x_min
    surface=largeur*profondeur
    
    return surface

#area in commun between tiny images generated by the filter and tiny images annoted
def area_intersection(x_min_gen,x_max_gen,y_min_gen,y_max_gen,  x_min_anote,x_max_anote,y_min_anote,y_max_anote   ):
    
    min_xmax=min(x_max_gen,x_max_anote)
    max_xmin=max(x_min_gen,x_min_anote)
    min_ymax=min(y_max_gen,y_max_anote) 
    max_ymin=max(y_min_gen,y_min_anote)    
    
    largeur=max(0,min_xmax-max_xmin)
    profondeur=max(0,min_ymax-max_ymin)
    area_intersection=largeur*profondeur
    return area_intersection


#extract x_mi,x_max,y_min,y_max
def get_table_coord(table_line):
    x_min=table_line["xmin"]
    y_min=table_line["ymin"]
    x_max=table_line["xmax"]
    y_max=table_line["ymax"]
    
    return x_min,x_max,y_min,y_max



    
#gather tiny images of 3Chanels in a unique array
def batched_cnts(cnts,imageB,imageSize= 28,filtre_color=False):
    
    #Initialisation de variables et de liste
    batchImages = []
    liste_table = []
    
    if filtre_color==False:
        for ic in range(0,len(cnts)):
            (x, y, w, h) = cv2.boundingRect(cnts[ic])
            f = pd.Series(dtype= "float64")
            f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
            subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            batchImages.append(subI)
            liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))     
    
    
    if filtre_color==True:
        
        for ic in range(0,len(cnts)):
            (x, y, w, h) = cv2.boundingRect(cnts[ic])
            f = pd.Series(dtype= "float64")
            f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
            subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
            subI = RecenterImage(subI,o)
            subI = cv2.resize(subI,(imageSize,imageSize))
            
            red=np.mean(subI[:,:,0])
            green=np.mean(subI[:,:,1])
            blue=np.mean(subI[:,:,2])
            colors=(red,green,blue)
            
            #or !=
            if np.max(colors)==blue:
                batchImages.append(subI)
                liste_table.append(np.array([ [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,4)))     
    
    
    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, imageSize,imageSize,3))
    table_non_filtre = pd.DataFrame(np.vstack(liste_table))
    table_non_filtre = table_non_filtre.rename(columns={ 0: 'xmin', 1: 'xmax', 2: 'ymin', 3: 'ymax'})
    table_non_filtre=table_non_filtre.astype(int)
    
    
    return batchImages_stack_reshape,table_non_filtre 

#Add a 4th chanel to all tiny images. This 4th chanels corresponding of the difference in HSV between to images compares 
def add_chanel(img,array_to_add):
    b_channel, g_channel, r_channel = cv2.split(img)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, array_to_add))
    
    return img_BGRA

#gather tiny images of 4Chanels in a unique array
def get_4C_all_batch(batchImages_stack_reshape,Diff,table_non_filtre):
    
      

    #Etape pour rajouter un canal sur chaque imagette
    #Diff=diff_filtre(imageA,imageB,method=diff_mod4C)
    imageSize=28
    subI_diff_liste=[]
    for i in range(len(table_non_filtre)):
        
        x_min, x_max, y_min, y_max=get_table_coord(table_non_filtre.iloc[i])
        
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x_min, x_max, y_min, y_max
        
        subI_diff, o, d, imageRectangles = GetSquareSubset(Diff,f,verbose=False)
        subi_expand=np.expand_dims(subI_diff,axis=2)
        subI_recenter = RecenterImage(subi_expand,o)
        subI_resize = cv2.resize(subI_recenter,(imageSize,imageSize))
        subI_resize=np.expand_dims(subI_resize,axis=2)
        subI_diff_liste.append(subI_resize)
    list_4C=list(map(add_chanel,batchImages_stack_reshape,subI_diff_liste))
    batchImages_stack_reshape=np.array(list_4C)
    
    return batchImages_stack_reshape








#When a tiny image is caught in the area of an annotation assigned it to a list corresponding in its category label
def assign_timages_to_annotation(generate_square,Images_target,dict_anotation_index_to_classe,precise=False):
    
    
    
    #initialisation du nv tableau
    Images_target_nv=Images_target.copy()
    Images_target_nv["nv_index"]=0
    Images_target_nv["nv_class"]=0
    nb_imagettes=len(Images_target)
    
    #set are of gen squares
    xmin_gen,xmax_gen,ymin_gen,ymax_gen=get_table_coord(generate_square)
    ln_square_gen=len(generate_square)
    generate_square["num_index"]=list(generate_square.index)
    generate_square["area"]=area_square(xmin_gen,xmax_gen,ymin_gen,ymax_gen)
    
    
    #get max intersection with square generate for each sqaure annotate
    for num_im in range(nb_imagettes):
        
        x_min_anote,x_max_anote,y_min_anote,y_max_anote=get_table_coord(Images_target.iloc[num_im])
        surface_anote=area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
        
        
        #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
        zip_xmin_anote=[x_min_anote]*ln_square_gen
        zip_xmax_anote=[x_max_anote]*ln_square_gen
        zip_ymin_anote=[y_min_anote]*ln_square_gen
        zip_ymax_anote=[y_max_anote]*ln_square_gen
        
        
        #Select the im generated with th maximum area in commun but not too big
        gen_square_size_filtered=generate_square[(generate_square["area"]<5*surface_anote) & (generate_square["area"]>0.5*surface_anote) ]
        xmin_gen,xmax_gen,ymin_gen,ymax_gen=get_table_coord(gen_square_size_filtered) 
        medium_squares=gen_square_size_filtered.copy()
        intersections_=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                            (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
        medium_squares.loc[:,"area_intersection"]=intersections_

 
        medium_squares=medium_squares[medium_squares["area_intersection"]>0.5*surface_anote]
        if len(medium_squares)!=0:
            max_squares=medium_squares[medium_squares["area_intersection"]==max(medium_squares["area_intersection"])]
            index_max_intersection=int(max_squares["num_index"][max_squares["area"]==min(max_squares["area"])])
            Images_target_nv["nv_index"].iloc[num_im]=index_max_intersection
            Images_target_nv["nv_class"].iloc[num_im]=dict_anotation_index_to_classe[Images_target_nv["classe"].iloc[num_im]]
            

    
    if precise==False:
        
        return Images_target_nv
    
    if precise==True:
        return Images_target_nv





#Count the number of objects inside the images, the images extracted, the difference generated. 
def count_extractions(generate_square,
                        Images_target,precise=False):
    
    
    
    an_caught=0
    NB_OBJECTS_TO_CAUGHT=len(Images_target)
    
    #set area of gen squares
    xmin_gen,xmax_gen,ymin_gen,ymax_gen=get_table_coord(generate_square)
    ln_square_gen=len(generate_square)
    generate_square["num_index"]=list(generate_square.index)
    generate_square["area"]=area_square(xmin_gen,xmax_gen,ymin_gen,ymax_gen)
    
    
    #get max intersection with square generate for each sqaure annotate
    for num_im in range(NB_OBJECTS_TO_CAUGHT):
        
        x_min_anote,x_max_anote,y_min_anote,y_max_anote=get_table_coord(Images_target.iloc[num_im])
        surface_anote=area_square(x_min_anote,x_max_anote,y_min_anote,y_max_anote)
        
        
        #Replicated the coordinates of annotations the number time of the len  to be able to apply area_intersection function
        zip_xmin_anote=[x_min_anote]*ln_square_gen
        zip_xmax_anote=[x_max_anote]*ln_square_gen
        zip_ymin_anote=[y_min_anote]*ln_square_gen
        zip_ymax_anote=[y_max_anote]*ln_square_gen
        
        
        #Select the im generated with th maximum area in commun but not too big
        gen_square_size_filtered=generate_square[(generate_square["area"]<5*surface_anote) & (generate_square["area"]>0.5*surface_anote) ]
        xmin_gen,xmax_gen,ymin_gen,ymax_gen=get_table_coord(gen_square_size_filtered) 
        medium_squares=gen_square_size_filtered.copy()
        liste_intersection=[area_intersection(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in zip 
                            (xmin_gen,xmax_gen,ymin_gen,ymax_gen, zip_xmin_anote,zip_xmax_anote,zip_ymin_anote,zip_ymax_anote ) ]
        
        medium_squares.loc[:,"area_intersection"]=liste_intersection
        #medium_squares=medium_squares.reset_index()
        
        
        
        
        medium_squares=medium_squares[medium_squares["area_intersection"]>0.5*surface_anote]
        if len(medium_squares)!=0:
            an_caught+=1
            

    
    return  an_caught,NB_OBJECTS_TO_CAUGHT,ln_square_gen




######################################################################################################################################################
##Hide unusefull part of the image

def set_mask(imageB,folder,number_chanels=3):
    
    masks_path=Mat_path+'masque/'
    map_masks={'timeLapsePhotos_Pi1_0': "mask_0.npy", 'timeLapsePhotos_Pi1_1': "mask_1.npy","timeLapsePhotos_Pi1_2": "mask_2.npy","timeLapsePhotos_Pi1_3": "mask_3.npy",  "timeLapsePhotos_Pi1_4": "mask_4.npy"}
    folder_num=folder.split("/")[2]
    

    print("masque activée")
    mask_path=masks_path+map_masks[folder_num]
    mask_image = load(mask_path)
    if number_chanels==3:
        ImageMaksed=np.multiply(imageB, mask_image) 
    elif number_chanels==1:
        ImageMaksed=np.multiply(imageB, mask_image[:,:,0]) 
    imageB=ImageMaksed.astype(int)
    
    return imageB



#order images names by time of shoot. Images should be ordered to compare the differences appear for all each couple of images compared.
def order_images(data_path):
    images_=[]
    for r, d, f in os.walk(data_path):
        for file in f:
            if '.jpg' in file:
                images_.append(basename(join(r, file)))  
    images_.sort()                   
    return images_   