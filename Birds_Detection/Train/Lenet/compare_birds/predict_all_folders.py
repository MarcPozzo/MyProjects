#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:09:54 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-

#L'objectif de ce script est de tester toutes les images avec les paramètres pour linstant meilleurs

#On va ensuite proposer les vrais positifs. Pour les images identifiées en tant que oiseaux mais aussi celles qui ne le sont pas



#Importation packages

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")

import functions_debug as fn
import pandas as pd
import os
from os.path import basename, join
import numpy as np

from keras.models import Model, load_model
import cv2
import ast       
            









# default parameters
fichierClasses= "Table_Labels_to_Class.csv" # overwritten by --classes myFile
Imagettes="imagettes.csv"
locImagettes="Rec_images"
#generatorFile="generateur_bigger.csv"
generatorFile="generateur.csv"
#fichierClasses = "classesAnimauxSemis.txt"


    






chdir('/mnt/VegaSlowDataDisk/c3po/Images_aquises')
frame=pd.read_csv(fichierClasses,index_col=False)



def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            yield from el
        else:
            yield el





def to_reference_labels (frame,df,class_colum):

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





#Maintenant je fais le dictionnaire label 0 donne "autre", 1 chevreuil
converion_nb_to_labels={}




#Paramètres à choisir

name_ref="image_2019-04-30_17-47-10.jpg"
coverage_threshold=0.5
#liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLAapsePhotos_Pi1_4'   ]
#Pour filtre quantile

neurone_features='/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob'


path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0','/DonneesPI/timeLapsePhotos_Pi1_1','/DonneesPI/timeLapsePhotos_Pi1_2','/DonneesPI/timeLapsePhotos_Pi1_3','/DonneesPI/timeLapsePhotos_Pi1_4'   ]

contrast=-5
blockSize=19
blurFact=15











coef_filtre=pd.read_csv("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/coefs_filtre_RQ.csv")

def birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,limit_area_square,contrast,neurone_features,blockSize,blurFact,folder,filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264):
    
    path="/mnt/VegaSlowDataDisk/c3po/Images_aquises"
    image_ref=path+path_images+name_ref
    image_test=path+path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    
    
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    
    #On enlève les éléménets inutiles, on peut aussi placer cel en dehors de la fonction pour que ça soit en dehors de la boucle
    
    imagettes=to_reference_labels (frame,imagettes,"classe")
    imagettes=imagettes[ (imagettes["classe"]!="abeille") & (imagettes["classe"]!='oiseau') & (imagettes["classe"]!='autre')
    & (imagettes["classe"]!='pie') & (imagettes["classe"]!='chat') 
    & (imagettes["classe"]!='sanglier') & (imagettes["classe"]!='cheval') & (imagettes["classe"]!='chat') ]

    folder_choosen="."+ folder
    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ]
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    #imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]
    imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
    if len(imagettes1["classe"].unique())>1:
        print("attention il y a des oiseaux de différents type ici le code n'est pas adapté à cette situation" )
    nom_classe=imagettes1["classe"].iloc[0]
    
    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=fn.filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast,blurFact=blurFact)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        #name = (os.path.split(name_test)[-1]).split(".")[0]
        #name = name + "_" + str(ic) + ".JPG"
        name = nom_classe + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = fn.GetSquareSubset(imageB,f,verbose=False)
        subI = fn.RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)


    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    
    table_quantile,index_possible_birds=fn.filtre_quantile(table_full,coef_filtre,height=2448,width=3264)
    table_filtre_RL=table_quantile.copy()
    table_filtre_RL["possible_bird"]=fn.filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
    p_bird=table_filtre_RL.index
    table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
    index_possible_birds=list(set(index_possible_birds).intersection(p_bird)) 
    #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_birds)]
    

    #Construisons les carrés pour les coordonnées issus de l'annotation 

    
    #Attention il manque la table
    #table_add=pd.read_csv(path_anotation)
    #annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    #annontation_reduit=annontation_reduit.iloc[::2]
    #annontation_reduit=im
    
    
    #On va essayer d'integre la table d'anotation qu'on pourra changer au grès de name_test
    




    #imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]=="image_2019-04-30_18-17-14.jpg"]
    
    to_drop=['path', 'filename', 'width', 'height', 'classe', 'index']

    im=imagettes1.drop(to_drop,axis=1)
    col=list(im.columns)
    col = col[-1:] + col[:-1]


    annontation_reduit=im[col]
    
    
    
    
    
    
    liste_carre_annote=annontation_reduit[['xmin', 'ymin', 'xmax', 'ymax']].apply(fn.to_polygon,axis=1)
    
    
    
    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    #generate_square=table.iloc[:,1:5]
    if filtre_choice=="No_filtre":
        generate_square=table_full.iloc[:,1:5]
    elif filtre_choice=="quantile_filtre":
        generate_square=table_quantile.iloc[:,1:5]
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL.iloc[:,1:5]
    
    liste_carre_diff=generate_square[['xmin', 'ymin', 'xmax', 'ymax']].apply(fn.to_polygon,axis=1)
    
    #Maintenant on va voir si les carrés des diffs ont suffisament de surfaces en commun avec les carrés des annotations

    
    #Initialisation
    liste_ANNOTATION=[]
    liste_DIFF=[]
    #nb_carre_diff=0
    nb_birds_match=0    

    proportion_limit=coverage_threshold
    
    


        
     
        
    for polygon_ref in liste_carre_annote:   
     
        max_proportion_test=0
        

        liste_des_max=[]
        #On va supprimer les carre_diff trop mais attention au carré
        
        liste_carre_diff_filtre=[i for i in liste_carre_diff if i.area < limit_area_square]
        
        for carre_diff in liste_carre_diff_filtre:
            intersection=polygon_ref.intersection(carre_diff)
            proportion=intersection.area/polygon_ref.area

            liste_des_max.append(proportion)
        print("la liste de controle est égale à:", len(liste_des_max))
        max_proportion_test=max(liste_des_max)
        position_maximum=np.argmax(liste_des_max)
        if (max_proportion_test>proportion_limit) :  
            liste_DIFF.append(position_maximum)
            nb_birds_match+=1
            #Au lieu de faire des +1 comme ça on va plutot lui demander la postion du max 
        
        
        
        
        
    nb_birds_to_find=len(liste_carre_annote)
    print( "nombre d'oiseau repérés", nb_birds_match)
    
    
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les diffs
        
    """ 
    imageRectangles=imageB.copy()
    #liste_DIFF=[499,610]
    
   
    for i in liste_DIFF:
        xmin=generate_square["xmin"].iloc[i]
        ymin=generate_square["ymin"].iloc[i]
        xmax=generate_square["xmax"].iloc[i]
        ymax=generate_square["ymax"].iloc[i]
        
        #cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les annontations
    
    #liste_ANNOTATION=[4,8]
    for i in liste_ANNOTATION:
        xmin=annontation_reduit["xmin"].iloc[i]
        ymin=annontation_reduit["ymin"].iloc[i]
        xmax=annontation_reduit["xmax"].iloc[i]
        ymax=annontation_reduit["ymax"].iloc[i]
    
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 
    #On va essayer d'écrites maintenant les carrés qui sont censés repérés les oiseaux et vérifier qu'ils matchent bien avec des oiseaux ( pas de décalage d'indices par exemple)
    
    
    #cv2.imwrite("Output_images/annotation_is_match.jpg",imageRectangles)
    """
    
    #En cas de zéro il faut soulever une erreur !!!
    liste_batch_images=range(len(batchImages_stack_reshape))
    liste_not_matche=set(liste_batch_images)-set(liste_DIFF)
    
    batchImages_match_birds = [batchImages_stack_reshape[i] for i in (liste_DIFF)]
    batchImages_match_not_birds = [batchImages_stack_reshape[i] for i in (liste_not_matche)]
    #
    try:
        batchImages_match_birds=np.vstack(batchImages_match_birds)
        batchImages_match_not_birds=np.vstack(batchImages_match_not_birds)
        #Prediction modèle
        model = load_model(neurone_features,compile=False)
        CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
        estimates_match_birds = CNNmodel.predict(batchImages_match_birds.reshape(-1,28,28,3))
        estimates_match_not_birds = CNNmodel.predict(batchImages_match_not_birds.reshape(-1,28,28,3))
        
    
        liste_prediction=list(estimates_match_birds.argmax(axis=1))
        TP=(len(liste_prediction)-liste_prediction.count(0))
        pourcentage_TP=TP/len(liste_prediction)
        
        
        
        liste_prediction_not_match=list(estimates_match_not_birds.argmax(axis=1))
        FP=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))
        pourcentage_FP=FP/len(liste_prediction_not_match)
        
        
        #On va tester une troisième liste, les very True Positif
        #Il faut être le bon type d'oiseaux
        #On retourn +1 si l'oiseau est bien classé, à la fin on regardera le chiffre finale on divisera eventuellement par le nombre d'oiseau prédits
        VTP=0
        for i in range(len(liste_prediction)):
            if liste_prediction[i]==nom_classe:
                VTP+=1
        
        #D'abord on va transformer estimates avec préféré le max
        #Dans le tableau il faudra convertir la classe string en classe nombre
        #Puis ensuite on se demande si les deux sont égaux
        
        #Il faut s'assurer que estimates est seulement pour les imagettes tout d'abord (puis ensuite on verra)
      
        if nb_birds_match>0:
            are_there_birds=True
        else:
            are_there_birds=False
        #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
    except ValueError:
        print("il n'a pas d'imagettes trouvées !")
        pourcentage_TP=0
        TP=0
        VTP=0
        pourcentage_FP=0
        FP=0
    return nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP






nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP=birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,
folder)




VTP=0

name_test="image_2019-04-30_18-17-14.jpg" 
name_ref="image_2019-04-30_18-16-57.jpg"





liste_folders=['/DonneesPI/timeLapsePhotos_Pi1_0'] 
Birds_well_predict_by_folder=[]

birds_predict_by_folder=[]
liste_Nb_oiseaux_a_reperer=[]
Nb_oiseaux_a_reperer_by_folder=[]
Nombre_imagette_oiseau_match_by_folder=[]
for folder in liste_folders:
    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
    
    chdir(path+folder)
    liste_image_ref = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path+folder):
        for file in f:
            if '.jpg' in file:
                liste_image_ref.append(basename(join(r, file)))
                
                
    path_images=folder+"/"
    folder_choosen="."+folder
    #imagettes_PI_0=imagettes[(imagettes["path"]=="./DonneesPI/timeLapsePhotos_Pi1_0") ]
    imagettes_PI_0=imagettes[(imagettes["path"]==folder_choosen) ]
    
    #Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer
    imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]



    Birds_well_predict=[]
    Nb_oiseaux_a_reperer=[]
    Nombre_imagette_oiseau_match=[]
    Pourcentage_birds_predict=[]
    Birds_predict=[]
    
    #Le code fait des répitions pour rien s'il y a plusieurs imagettes et Nombre_imagette_oiseau
    dict_images_catched={}
    liste_name_test=list(imagettes_PI_0["filename"].unique())
    for name_test in liste_name_test[0:2]:
        index_of_ref=liste_image_ref.index(name_test)-1
        name_ref=liste_image_ref[index_of_ref]
        print(name_test,name_ref)
        #catched_bird=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000)
        #dict_images_catched[name_test]=catched_bird
        #nombre_imagette_oiseau_match,pourcentage_TP,nb_oiseaux_a_reperer,TP=birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        
        nb_birds_to_find,nb_birds_match,pourcentage_TP,pourcentage_FP,TP,FP,VTP=birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize,blurFact,folder)
        if VTP>0:
            print("VTP supérieur à 0")
            print("VTP supérieur à 0")
            print("VTP supérieur à 0")
            print("VTP supérieur à 0")
        else:
            print("pas de problème")
        Pourcentage_birds_predict.append(pourcentage_TP)
        Nombre_imagette_oiseau_match.append(nb_birds_match)
        Nb_oiseaux_a_reperer.append(nb_birds_to_find)
        Birds_predict.append(TP)
        Birds_well_predict.append(VTP)
    #print(Birds_predict)
    Nb_oiseaux_a_reperer_by_folder.append(sum(Nb_oiseaux_a_reperer))
    print(sum(Birds_predict))
    print(folder)
    birds_predict_by_folder.append(sum(Birds_predict))   
    Nombre_imagette_oiseau_match_by_folder.append(sum(Nombre_imagette_oiseau_match))
    Birds_well_predict_by_folder.append(sum(Birds_well_predict))

print("Nb_oiseaux_a_reperer_by_folder",Nb_oiseaux_a_reperer_by_folder)
print("Nombre_imagette_oiseau_match_by_folder",Nombre_imagette_oiseau_match_by_folder)
print("birds_predict_by_folder",birds_predict_by_folder)
print("Birds_well_predict by folder",Birds_well_predict_by_folder)


"""
Resulat pour les 30 premiers éléments de chaque dossier

Nb_oiseaux_a_reperer_by_folder [56, 41, 31, 48, 47]
Nombre_imagette_oiseau_match_by_folder [56, 39, 30, 41, 34]
birds_predict_by_folder [50, 12, 20, 34, 20]
Birds_well_predict by folder [51, 18, 20, 36, 19]

Conclusion partielle
Pour les deux derniers dossier le filtre ne matche pas bien les oiseaux
Les réusltats en terme de prédiction sont particulièrement mauvais pour le deuxième fichier 1/3 d'oiseau prédit
#Il faudrait regarder le deuxième dossier ou bien le dernier
"""
