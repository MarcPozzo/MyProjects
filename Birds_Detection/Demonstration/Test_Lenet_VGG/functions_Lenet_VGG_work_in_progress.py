#This script is the support when it comes to test Lenet or VGG16
#The script is divided in several sections noted with ##:Predictions, Filters functions, Evaluate predictions, 
#Assign a class to an image, Adjust extracted imagette, Add 4th chanel
#import imutils



import ast
import cv2
import pandas as pd
import numpy as np
from numpy import load
from imutils import grab_contours
#from skimage.measure import compare_ssim
import math
#from keras.applications import VGG16
import joblib


#Load models
Mat_path="../../Materiels/"
Mod_path=Mat_path+"Models/"
Model1 = joblib.load(Mod_path+"model.cpickle")
filtre_RL = joblib.load(Mod_path+"RL_annotation_model")
coef_filtre=pd.read_csv(Mod_path+"coefs_filtre_RQ.csv")
#model = VGG16(weights="imagenet", include_top=False)

######################################################################################################################################################

##Prediction

#peut être enlever la partie 4C
#Prediction with Lenet with 3 or 4chanels for Lenet neural networks
#Peut être renomer Lenet_prediction_evaluation
#Reorganiser para
#Soit on garde les para et on les enlève à côté ou bien, on les enlève ici ...
"""def Lenet_prediction(name_test,name_ref,folder,CNNmodel,maxAnalDL,seuil=210,
                 diff_mod="HSV",method="light",
                 chanels=3,numb_classes=6,mask=False,coverage_threshold=0.99,
                 contrast=-5,blockSize=53,blurFact=15,
                 filtre_choice="No_filtre", thresh=0.5, thresh_active=True,index=False,
                 down_thresh=25,focus="bird_prob"):"""


def  Evaluate_Lenet_prediction ( Images , name_test , name_ref  , CNNmodel , maxAnalDL , seuil = 210 ,data_path='../../../../Pic_dataset/',
                 diff_mod = "HSV" , method = "light" ,
                 chanels = 3 , numb_classes = 6 , mask = False , coverage_threshold = 0.99 ,
                 contrast = - 5 , blockSize = 53 , blurFact = 15 ,
                 filtre_choice = "No_filtre" , thresh = 0.5 , thresh_active = True , index = False ,
                 down_thresh = 25 , focus = "bird_prob" ):
    
    #Faire un seuil à 29 de surface
    #Parameters
    #Dictionnary to convert string labels to num labels
    dic_labels_to_num,dic_num_to_labels=dictionnaire_conversion_mclasses(numb_classes)
    
    #Definition des images
    data_path='../../../../Pic_dataset/'
    image_ref=data_path+name_ref
    image_test=data_path+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    Images_target=Images[Images["filename"]==name_test]
    #A-t-on besoin d'enlever filename
    Images_target=Images_target.drop('filename',axis=1)
    nb_oiseaux=len( Images_target)
    print("Birds in the picture: ",nb_oiseaux)
    #Set Mask
 
    """
    if mask==True:
        imageA=mask_function_bis(folder,imageA,number_chanels=3)
        imageB=mask_function_bis(folder,imageB,number_chanels=3)
        imageB=imageB.astype(np.uint8)
        imageA=imageA.astype(np.uint8)
    """   
  
    cnts=diff_images(imageA,imageB,contrast=contrast,blockSize=blockSize,blurFact = blurFact,method=method,seuil=seuil)

    if  cnts :
        batchImages_stack_reshape,table_non_filtre=batched_cnts(cnts,imageB) 
        generate_square,batchImages_stack_reshape=filtre_line_bis(table_non_filtre,filtre_choice,batchImages_stack_reshape,down_thresh)
        if len(generate_square)!=0:
            batchImages_stack_reshapes,generate_square=pre_select_bis(batchImages_stack_reshape,generate_square,cnts,maxAnalDL=maxAnalDL)
    
        
            
        #Add a 4th chanel (differention with the previous image)
        #C est plutot sur generate_square qu'il faudrait
        if chanels==4:
            Diff=diff_filtre(imageA,imageB,method=diff_mod)
            batchImages_stack_reshape=get_4C_all_batch(batchImages_stack_reshape,Diff,table_non_filtre)
    
        #On classe les imagettes génères en fonction de l'espace avec laquelle ses coordonnees correspondent sur l'image
        #Il faudrait vérifier ici qu'on est vraiment obligé de bouger les colonnes comme on l'a fait au-dessus
        (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,liste_DIFF_corbeau,
         liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)=class_imagettes_sans_dboucle(generate_square,coverage_threshold, Images_target,dic_labels_to_num) 
        
        (liste_Diff_birds,liste_Diff_animals,birds_match,liste_Diff_not_birds,liste_Diff_animals,
        liste_DIFF_not_matche)=rearrange_dif(liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,birds_defined_match,batchImages_stack_reshape)
        
    else:
        batchImages_stack_reshape=[]
        generate_square=[]
    #Estimations established and classed
    if batchImages_stack_reshape!=[]:
        estimates = CNNmodel.predict(np.array(batchImages_stack_reshape))
        #estimates = CNNmodel.predict(batchImages_stack_reshape)
        #predictions=list(estimates.argmax(axis=1))
        TP_birds,FP,TP_estimates,FP_estimates= miss_well_class(estimates,liste_Diff_birds,liste_DIFF_not_matche,
                        liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_Diff_animals,
         
               thresh_active,thresh,numb_classes=6,focus=focus,index=index)
    else:
        TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds=[[] for i in range(5)]
  
    return imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates,liste_Diff_birds,nb_oiseaux








#Combine VGG16 outputs from pre-trained weight and logistic trained with the output of VGG16
def stacking_prediction(name_test,name_ref,folder,
                                 method="ssim",numb_classes=2,mask=True,coverage_threshold=0.99,contrast=-5,blockSize=53,blurFact=25,filtre_choice="No_filtre", thresh=0.99, thresh_active=True,index=False,down_thresh=25):

    

    dic_labels_to_num= {'no_animal': 0,'animal': 1,'autre': 0,'chevreuil': 1,'corneille': 2,'faisan': 3,'lapin': 4,'pigeon': 5, 'oiseau': 6}
 
    
    #Definition des images
    path_images=folder+"/"
    path="/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises"

    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    #Ouverture des fichiers annot�s  si on voulait gagner du temps on pourrait le sortir de la fonction
    imagettes=pd.read_csv("/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po/Images_aquises/imagettes.csv")
    nom_classe,imagettes_target=open_imagettes_file(imagettes,folder,name_test)


    #Set Mask
    if mask==True:
        imageA=mask_function_bis(folder,imageA,number_chanels=3)
        imageB=mask_function_bis(folder,imageB,number_chanels=3)
        imageB=imageB.astype(np.uint8)
        imageA=imageA.astype(np.uint8)
  
        
    #Organize generated imagettes and apply filters
    #cnts=filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    #cnts=filtre_khalid(imageA,imageB)
    cnts=diff_images(imageA,imageB,blurFact = blurFact,method=method)
    if cnts!=[]:
        batchImages_stack_reshape,table_non_filtre=batched_cnts(cnts,imageB,224) 
        generate_square,batchImages_stack_reshape=filtre_line_bis(table_non_filtre,filtre_choice,batchImages_stack_reshape,down_thresh)
        batchImages_stack_reshape=np.array(batchImages_stack_reshape)
    else:
        batchImages_stack_reshape=np.zeros(shape=(0,0))
        generate_square=0
 
    #Estimations established and classed
    if  batchImages_stack_reshape.size :
       

        #start=time.time()
        #estimates = model.predict(np.array(batchImages_stack_reshape))
        maxAnalDL=len(generate_square)
        #max_probas=filtre_RL.predict_proba(np.array(generate_square, dtype = "float64"))[:,0]
        max_probas=filtre_RL.predict_proba(np.array(generate_square.iloc[:,1:], dtype = "float64"))[:,0]
        max_probas = np.argsort(-max_probas)
        #c'est ici le problème à mon avis

        #maxAnalDL=12
        index_filter=list(max_probas)[:min(len(cnts),maxAnalDL)]
        generate_square = generate_square.iloc[index_filter,:]
        batchImages_stack_reshape = [batchImages_stack_reshape[i] for i in index_filter]
        generate_square=generate_square.loc[index_filter]
        generate_square=generate_square.reset_index()
        
        #batchImages_stack_reshape = [batchImages_stack_reshape[i] for i in [41,28]]

        
        features = model.predict(np.array(batchImages_stack_reshape), batch_size=4) # why 5 if only 4 proc on the pi...
        features = features.reshape((features.shape[0], 7 * 7 * 512))

        predictions = Model1.predict(features)
        estimates = list(predictions)
        
        
        
        (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,liste_DIFF_corbeau,
         liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)=class_imagettes_sans_dboucle(generate_square,coverage_threshold,imagettes_target,dic_labels_to_num) 
    
        (liste_Diff_birds,liste_Diff_animals,birds_match,liste_Diff_not_birds,liste_Diff_animals,
         liste_DIFF_not_matche)=rearrange_dif(liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,birds_defined_match,batchImages_stack_reshape)
        
        
        TP_birds,FP,TP_estimates,FP_estimates= miss_well_class(estimates,liste_Diff_birds,liste_DIFF_not_matche,
                        liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_Diff_animals,thresh_active,thresh,numb_classes=2,focus="animals",index=True)
        print(TP_birds)

               
    else:
        TP_birds,FP,TP_estimates,FP_estimates=[[] for i in range(4)]
  
    return imageA,imageB,cnts,batchImages_stack_reshape,generate_square,TP_birds,FP,TP_estimates,FP_estimates





######################################################################################################################################################
##Filters functions
    
def diff_images(imageA,imageB,contrast=-5,blockSize=51,blurFact = 25,method="light",seuil=210):
    
    if method=="light":
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
    

        
    elif method=="ssim":
        
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





def diff_filtre(imageA,imageB,method="HSV"):
    if method=="HSV":
        img2 = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
        img1 = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
        absDiff2 = cv2.absdiff(img1, img2)
        diff_Gray = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        
    elif method=="BGR":
        absDiff2 = cv2.absdiff(imageA, imageB)
        diff_Gray = cv2.cvtColor(absDiff2, cv2.COLOR_BGR2GRAY)
        
    else:
        print("erreur de saisie de la methode")
        
    return diff_Gray




#return  only squares enought small to be birds
def filtre_line_bis(table_non_filtre,filtre_choice,batchImages_stack_reshape,down_thresh):

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






def pre_select_bis(batchImages_stack_reshape,generate_square,cnts,maxAnalDL=12):
    
    if maxAnalDL>=1:
        
     
        max_probas=filtre_RL.predict_proba(np.array(generate_square.iloc[:,1:], dtype = "float64"))[:,0]
        max_probas = np.argsort(-max_probas)
       
          
        index_filter=list(max_probas)[:min(len(cnts),maxAnalDL)]
        generate_square = generate_square.iloc[index_filter,:]
        batchImages_stack_reshape = [batchImages_stack_reshape[i] for i in index_filter]
        generate_square=generate_square.loc[index_filter]
        generate_square=generate_square.reset_index()
        
    elif 0<maxAnalDL<1:
      
        max_probas=filtre_RL.predict_proba(np.array(generate_square.iloc[:,1:], dtype = "float64"))[:,0]
        index_filter=np.where(max_probas>maxAnalDL)[0].tolist()
          
        generate_square = generate_square.iloc[index_filter,:]
        batchImages_stack_reshape = [batchImages_stack_reshape[i] for i in index_filter]
        generate_square=generate_square.loc[index_filter]
        generate_square=generate_square.reset_index()
        
    else:
        (batchImages_stack_reshape,generate_square)=(batchImages_stack_reshape,generate_square)
        
    return batchImages_stack_reshape,generate_square


######################################################################################################################################################
##Evaluate predictions


def miss_well_class(estimates,liste_Diff_birds,liste_DIFF_not_matche,
                    liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_Diff_animals,
                    thresh_active,thresh,numb_classes=6,focus="bird_large",index=False):
    
    TP_estimates=[]
    FP_estimates=[]
    TP_thresh_index=[]
    FP_thresh_index=[]
    if type(estimates)!=list:
        liste_prediction=list(estimates.argmax(axis=1))
    else:
        liste_prediction=estimates
        
    
    if focus!="bird_prob":
        index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon= class_predictions_dictionnaire(liste_prediction,numb_classes)
        index_animals=index_birds+index_other_animals
        
    elif focus=="bird_prob":
        Estimates_Birds_list=[]
        for i in range(len(estimates)):
            estimates_birds=estimates[i][2]+estimates[i][3]+estimates[i][5]
            Estimates_Birds_list.append(estimates_birds)
        birds_array=np.argwhere(np.array(Estimates_Birds_list)>thresh).tolist()         
        index_birds = [item for sublist in birds_array for item in sublist]  
        
        others_pr_birds=set(liste_DIFF_not_matche).intersection(index_birds)
        birds_pr_birds=set(liste_Diff_birds).intersection(index_birds)
        TP=birds_pr_birds
        FP=others_pr_birds
        
        #ici il faut faire 
        FP_thresh_index=[i for i in FP if max(estimates[i])>thresh]
        TP_thresh_index=[i for i in TP if max(estimates[i])>thresh]
        FP_birds_estimates=[max(estimates[i]) for i in (others_pr_birds)]
        birds_pr_bird_estimates=[max(estimates[i]) for i in birds_pr_birds]
        TP_estimates=birds_pr_bird_estimates
        FP_estimates=FP_birds_estimates
        #liste_of_all=[i for i in range(len(estimates))]
        #index_others= list(set(liste_of_all) - set(index_birds))
        
    
    if focus=="bird_large":
        others_pr_birds=set(liste_DIFF_not_matche).intersection(index_birds)
        birds_pr_birds=set(liste_Diff_birds).intersection(index_birds)
        TP=birds_pr_birds
        FP=others_pr_birds
        
        #ici il faut faire 
        FP_thresh_index=[i for i in FP if max(estimates[i])>thresh]
        TP_thresh_index=[i for i in TP if max(estimates[i])>thresh]
        FP_birds_estimates=[max(estimates[i]) for i in (others_pr_birds)]
        birds_pr_bird_estimates=[max(estimates[i]) for i in birds_pr_birds]
        TP_estimates=birds_pr_bird_estimates
        FP_estimates=FP_birds_estimates
        
    if focus=="bird_precise":
        TP_birds=list(set(liste_DIFF_corbeau).intersection(index_corbeau))+list(set(liste_DIFF_faisan).intersection(index_faisan))+list(set(liste_DIFF_pigeon).intersection(index_pigeon))
        TP=TP_birds 
        print("attention on a pas encore regl� FP")
        
        TP_birds_estimates=[estimates[i] for i in (TP_birds)]
        TP_estimates=TP_birds_estimates
        FP_estimates=[max(estimates[i]) for i in (FP)]
        
    if focus=="animals":
        FP_animals=set(liste_DIFF_not_matche).intersection(index_animals)
        animals_predict=set(liste_Diff_animals).intersection(index_animals)
        TP=animals_predict
        FP=FP_animals
        
        if numb_classes!=2:
            other_animals_estimates = [max(estimates[i]) for i in (index_other_animals)]
            FP_estimates=other_animals_estimates
            
            animals_match_estimates=[max(estimates[i]) for i in liste_Diff_animals]
            TP_estimates=animals_match_estimates
            
    if thresh_active==True and numb_classes!=2:


        TP_estimates=[i for i in TP_estimates if i > thresh]
        FP_estimates=[i for i in FP_estimates if i > thresh]
        
        
    if index==False:
        return TP,FP,TP_estimates,FP_estimates
    if index==True:
        return TP,FP,TP_thresh_index,FP_thresh_index  




def class_predictions_dictionnaire(liste_prediction,class_num): 
   #liste_DIFF_faisan,liste_DIFF_corbeau,liste_DIFF_pigeon,liste_DIFF_lapin,liste_DIFF_chevreuil,liste_DIFF_birds_undefined = ([] for i in range(6))
   
   index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon = ([] for i in range(8))
   

   map_indexes_2classes={"1":index_others, "0" :index_birds}
   map_indexes_6classes={"0":index_others, "1" :index_chevreuil,"2":index_corbeau,"3":index_lapin,"4":index_faisan,"5":index_pigeon }
   map_indexes_8classes={"0":index_others, "1" :index_chevreuil,"3":index_corbeau,"4":index_lapin,"5":index_faisan,"6":index_pigeon,"2":index_others,"7":index_others }
   
   if class_num==2:
       map_indexes=map_indexes_2classes
   elif class_num==6:
       map_indexes=map_indexes_6classes
   elif class_num==8:
       map_indexes=map_indexes_8classes
   else:
       print("le nombre de classes est mal spécifiée")
    
   for i, j in enumerate(liste_prediction):
       #print(i,j)
       map_indexes[str(j)].append(i) 
    
    
   index_other_animals=index_chevreuil+index_lapin
   if     class_num !=2:
       index_birds=index_corbeau+index_faisan+index_pigeon
   
   return index_others,index_birds,index_other_animals,index_chevreuil,index_corbeau,index_lapin,index_faisan,index_pigeon

#Rearange categories of images annoted matched by the camera 
def rearrange_dif(liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,
                  birds_defined_match,batchImages_stack_reshape):
    
    #Make larger categories
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    
    #Not matched categories
    liste_batch_images=range(len(batchImages_stack_reshape))
    liste_Diff_not_birds=set(liste_batch_images)-set(liste_Diff_birds)
    liste_DIFF_not_matche=set(liste_batch_images)-set(liste_Diff_birds)-set(liste_DIFF_other_animals)
    
    birds_match=birds_defined_match+len(liste_DIFF_birds_undefined)
    print("nombre d'oiseaux captur�s", birds_match)

    return liste_Diff_birds,liste_Diff_animals,birds_match,liste_Diff_not_birds,liste_Diff_animals,liste_DIFF_not_matche




#class imagettes if strict condition on area square and intersections

def class_imagettes_sans_dboucle(generate_square,coverage_threshold,
                        imagettes_target,dic_labels_to_num,precise=False):
    
    #Initialize empty list and dictionnary
    liste_DIFF_birds_defined,liste_DIFF_birds_undefined,liste_DIFF_other_animals,liste_DIFF_faisan,liste_DIFF_corbeau,liste_DIFF_pigeon,liste_DIFF_lapin,liste_DIFF_chevreuil=([] for i in range(8))
    dict_anotation_index_to_classe={}
    
    map_classes={"faisan":liste_DIFF_faisan, "corneille" : liste_DIFF_corbeau,"pigeon":liste_DIFF_pigeon,
                 "lapin" :liste_DIFF_lapin, "chevreuil" :liste_DIFF_chevreuil, "oiseau" : liste_DIFF_birds_undefined,
                 "incertain": liste_DIFF_birds_undefined, "pie":liste_DIFF_birds_undefined }
    
    
    
    birds_liste=["corneille","pigeon","faisan","oiseau","pie","incertain"]
    nb_imagettes=len(imagettes_target)
    
    #set are of gen squares
    xmin_gen,xmax_gen,ymin_gen,ymax_gen=get_table_coord(generate_square)
    ln_square_gen=len(generate_square)
    generate_square["num_index"]=list(generate_square.index)
    generate_square["area"]=area_square(xmin_gen,xmax_gen,ymin_gen,ymax_gen)
    
    
    #get max intersection with square generate for each sqaure annotate
    for num_im in range(nb_imagettes):
        
        classe=imagettes_target["classe"].iloc[num_im]
        x_min_anote,x_max_anote,y_min_anote,y_max_anote=get_table_coord(imagettes_target.iloc[num_im])
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
            max_squares=medium_squares[medium_squares["area_intersection"]==max(medium_squares["area_intersection"])]
            index_max_intersection=int(max_squares["num_index"][max_squares["area"]==min(max_squares["area"])])
            #index_max_intersection=int(medium_squares["num_index"][medium_squares["area_intersection"]==max(medium_squares["area_intersection"])])
            
      
            #ici if pour dic
            map_classes[classe].append(index_max_intersection)   
            
            dict_anotation_index_to_classe[str(index_max_intersection)]=dic_labels_to_num[classe]
    
    #Rec indexes of selected imagette in appropriate list         
    liste_DIFF_other_animals=liste_DIFF_chevreuil+liste_DIFF_lapin        
    liste_DIFF_birds_defined=liste_DIFF_faisan+liste_DIFF_corbeau+liste_DIFF_pigeon
    liste_Diff_birds=liste_DIFF_birds_defined+liste_DIFF_birds_undefined
    liste_Diff_animals=liste_Diff_birds+liste_DIFF_other_animals
    
    
    #Display the number of anomals and catched animals in this picture
    liste_DIFF_other_animals=liste_DIFF_chevreuil+liste_DIFF_lapin
    birds_table=imagettes_target.isin(birds_liste)
    nb_animals_match=len(liste_DIFF_birds_defined)+len(liste_DIFF_birds_undefined)+len(liste_DIFF_other_animals)
    nb_animals_to_find=len(imagettes_target)
    birds_to_find=len(birds_table)
    birds_defined_match=len(liste_DIFF_birds_defined)
    print("nombre d'oiseaux dans l'image",birds_to_find)
    print("nombre d'oiseaux repérés parmi les oiseaux lab�lis�s",birds_defined_match )
    
    if precise==False:
        
        return (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,
    liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_other_animals)
    
    if precise==True:
        return (liste_Diff_animals,dict_anotation_index_to_classe,liste_DIFF_birds_defined,liste_DIFF_birds_undefined,birds_defined_match,
    liste_DIFF_corbeau,liste_DIFF_faisan,liste_DIFF_pigeon,liste_DIFF_lapin,liste_DIFF_chevreuil)


######################################################################################################################################################
##Assign a class to an image


#Transform labels to new one already known
def to_reference_labels (df,class_colum,frame):

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


def dictionnaire_conversion_mclasses(numb_classes):
    dic_labels_to_num={}
    dic_num_to_labels={}
    
    liste_8_classes=["arbre","chevreuil","ciel","corneille","faisan","lapin","pigeon","terre","oiseau"]
    liste_6_classes=["autre","chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
    liste_2_classes=["no_animal","animal"]
    
    if numb_classes==6:
        liste_classes=liste_6_classes
    elif numb_classes==8:
        liste_classes=liste_8_classes
    elif numb_classes==2:
        liste_classes=liste_2_classes
        
    else :
        print("le dictionnaire pour ce nombre de classe n'est pas renseigné")
    for i, j in enumerate(liste_classes):
        dic_labels_to_num[j]=i
        dic_num_to_labels[i]=j

    
    return dic_labels_to_num,dic_num_to_labels

#On devrait l'inserer directement dans les fichiers de prediction
#Il y en a pour 3 lignes ...
def open_imagettes_file(Images,folder,name_test):
    
    """
    #Select only animals categories
    liste_to_keep=["chevreuil","corneille","faisan","lapin","pigeon","oiseau"]
    imagettes=to_reference_labels (imagettes,"classe")
    imagettes=imagettes[imagettes["classe"].isin(liste_to_keep)]    
    
    
    folder_choosen="."+ folder
    imagettes_folder=imagettes[(imagettes["path"]==folder_choosen) ]
    """
    

    #On selectionne seulement pour la photo sur laquel on veut rep�rer les oiseaux ou autres animaux et on r�arange les colonnes dans le bon ordre
    Images_target=Images[Images["filename"]==name_test]
    to_drop=['filename']
    Images_target=Images_target.drop(to_drop,axis=1)
    col = list(Images_target.columns)[-1:] + list(Images_target.columns)[:-1]
    Images_target=Images_target[col]
    
    
    
    #On regarde si il y a des imagettes de type diff�rents pas forc�ment utiles surtout si on garde les oiseaux undefined
    if len(Images_target["classe"].unique())>1:
        print("attention il y a plusieurs especes d'animaux" )
    #nom_classe=imagettes1["classe"].iloc[0]
    nom_classe=list(Images_target["classe"].unique())
    
    return nom_classe,Images_target





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


######################################################################################################################################################
##Adjust extracted imagette

#Poue avoir un resultat en 3�me dim ou en 1 dim
def RecenterImage(subI,o):
    
    h,l,r=subI.shape #on sait que r=3 (3 channels)
    
    # add to the dimension the dimensions of the cuts (due to image borders)
    h = h + o.ymincut + o.ymaxcut
    l = l + o.xmincut + o.xmaxcut

    t= np.full((h, l, r), fill_value=int(round(subI.mean())),dtype=np.uint8) # black image the size of the final thing

    t[o.ymincut:(h-o.ymaxcut),o.xmincut:(l-o.xmaxcut)] = subI
    
    return t




def GetSquareSubset(img,h,verbose=True, xml = False):
    
   
    # d�termine le plus grand c�te du carr�
    d = max(h.ymax-h.ymin,h.xmax-h.xmin)
      
    # d�termine le centre du carr�
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
        # d�ssine le carr�
        cv2.rectangle(img, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #�criture des images
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


 
    
  
# Verifie les bords de l'image pour la d�coupe    
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




#extract x_mi,x_max,y_min,y_max
def get_table_coord(table_line):
    x_min=table_line["xmin"]
    y_min=table_line["ymin"]
    x_max=table_line["xmax"]
    y_max=table_line["ymax"]
    
    return x_min,x_max,y_min,y_max





def area_square(x_min,x_max,y_min,y_max):
    

    
    profondeur=y_max-y_min
    largeur=x_max-x_min
    surface=largeur*profondeur
    
    return surface


def area_intersection(x_min_gen,x_max_gen,y_min_gen,y_max_gen,  x_min_anote,x_max_anote,y_min_anote,y_max_anote   ):
    
    min_xmax=min(x_max_gen,x_max_anote)
    max_xmin=max(x_min_gen,x_min_anote)
    min_ymax=min(y_max_gen,y_max_anote) 
    max_ymin=max(y_min_gen,y_min_anote)    
    
    largeur=max(0,min_xmax-max_xmin)
    profondeur=max(0,min_ymax-max_ymin)
    area_intersection=largeur*profondeur
    return area_intersection


def mask_function_bis(folder,imageB,number_chanels=3):
    
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

######################################################################################################################################################
##Add 4th chanel
    
def get_4C_all_batch(batchImages_stack_reshape,Diff,table_non_filtre):
    
      

    #Etape pour rajouter un canal sur chaque imagette
    #Diff=diff_filtre(imageA,imageB,method=diff_mod)
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

def add_chanel(img,array_to_add):
    b_channel, g_channel, r_channel = cv2.split(img)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, array_to_add))
    
    return img_BGRA
