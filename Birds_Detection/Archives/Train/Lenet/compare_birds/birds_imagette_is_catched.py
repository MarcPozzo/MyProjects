#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:45:28 2020

@author: pi




"""
#Ce script propose de vérifier si les carrés d'annotations sont bien repérées par la différence puis après le ou les filtres sous forme de fonction avec un script sourçale et sunthétique


#Pour l'instant les essais pour les bases pi ne sont pas trop conclants
#Il semblerait que le contraste entre la couleur des oiseaux et le champs ne permet pas de les distinguer mais de distinguer simplement leur ombre
#Il faudrait mettre de l'ordre dans le script pour qu'on est à mettre que de paramètres et que le reste s'execute
#Verifier que c'est les deux bonnes photos qui se comparent mais à priori oui vu le peu de cnts.
#On peut aussi tester avec la photo précédante comme dans boucle et non pas une photo de ref où il n'y a pas d'oiseau

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
#import functions as fn
exec(open("functions.py").read())
import cv2
import pandas as pd
import joblib
import matplotlib.pyplot as plt

output_Images_path="/mnt/VegaSlowDataDisk/c3po_interface/bin/Output_images/"

#Paramètres à choisir

#Pour filtre quantile



filtre_choice="No_filtre" #"quantile_filtre"#"No_filtre" "RL_filtre"



#Autres Paramètres
path="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/"

c3poFolder="/mnt/VegaSlowDataDisk/c3po_interface/"
#Model1 = joblib.load(c3poFolder+"bin/output/model.cpickle")
filtre_RL = joblib.load(c3poFolder+"bin/output/RL_annotation_model")



coef_filtre=pd.read_csv("testingInputs/coefs_filtre_RQ.csv")

name2 = "EK000228.JPG"
imageA = cv2.imread("testingInputs/EK000227.JPG")
imageB = cv2.imread("testingInputs/"+name2)
 


#Neural_models=["zoom_0.9:1.3_flip","6c_rob","zoom_1.3","drop_out.50","z1.3"]
name_model="z1.3"
neurone_features=path+name_model
path_anotation="testingInputs/oiseau_lab_Alex.csv"




#print(fn.birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2)==True)










#pb avec /mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_09-35-31.jpg c'est tout vert
# reflet /mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_3/image_2019-05-29_07-55-02.jpg

#essayons avec 
#Corbeaux








path_image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-27-49.jpg"

path_image_ref="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-24-28.jpg"



imageA = cv2.imread(path_image_ref)
imageB = cv2.imread(path_image_test)




a='./DonneesPI/timeLapsePhotos_Pi1_0' 
b='./DonneesPI/timeLapsePhotos_Pi1_1' 
c='./DonneesPI/timeLapsePhotos_Pi1_2' 
d='./DonneesPI/timeLapsePhotos_Pi1_3' 
e='./DonneesPI/timeLAapsePhotos_Pi1_4'

imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")

imagettes_PI_0=imagettes[(imagettes["path"]==a) ]


name_test="image_2019-04-30_18-27-49.jpg"
#imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]=="image_2019-04-30_18-17-14.jpg"]
imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
to_drop=['path', 'filename', 'width', 'height', 'classe', 'index']

im=imagettes1.drop(to_drop,axis=1)
col=list(im.columns)
col = col[-1:] + col[:-1]


annontation_reduit=im[col]

#annontation_reduit=im






cnts=filtre_light(imageA,imageB)

#cnts=filtre_light(img_test,img_ref)

#Essayons de refaire le filtre pour comprendre si qqc ne fonctionne pas


name_test="image_2019-04-30_18-27-49.jpg"

#def birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2,name_test,height=2448,width=3264):

def birds_is_catched(imageA,imageB,name2,neurone_features=neurone_features,filtre_choice="No_filtre",coef_filtre=coef_filtre,height=2448,width=3264):


    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=filtre_light(imageA,imageB)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name2)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
        f = pd.Series(dtype= "float64")
        f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif

        subI, o, d, imageRectangles = GetSquareSubset(imageB,f,verbose=False)
        subI = RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        batchImages.append(subI)
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
     
    table_full = pd.DataFrame(np.vstack(liste_table))
    
    
    #Ce serait bien de rajouter rename ici !!! Si ça n'entraine pas de bug
    table_full = table_full.rename(columns={0: 'imagettename', 1: 'xmin', 2: 'xmax', 3: 'ymin', 4: 'ymax'})
    table_full.iloc[:,1:]=table_full.iloc[:,1:].astype(int)


    batchImages_stack = np.vstack(batchImages)
    batchImages_stack_reshape=batchImages_stack.reshape((-1, 28,28,3))
    """
    table_quantile,index_possible_birds=filtre_quantile(table_full,coef_filtre,height=2448,width=3264)
    table_filtre_RL=table_quantile.copy()
    table_filtre_RL["possible_bird"]=filtre_RL.predict(np.array(table_filtre_RL.iloc[:,1:]))
    table_filtre_RL=(table_filtre_RL[table_filtre_RL["possible_bird"]=="O"])
    p_bird=table_filtre_RL.index
    table_filtre_RL.drop("possible_bird",axis=1,inplace=True)     
    index_possible_birds=list(set(index_possible_birds).intersection(p_bird)) 
    #batchImages_filtre = [batchImages_stack_reshape[i] for i in (index_possible_birds)]
    """

    #Construisons les carrés pour les coordonnées issus de l'annotation 

    
    #Attention il manque la table
    """table_add=pd.read_csv(path_anotation)
    annontation_reduit=(table_add.iloc[:,6:12]).drop("index",axis=1)
    annontation_reduit=annontation_reduit.iloc[::2]"""
    #annontation_reduit=im
    
    
    #On va essayer d'integre la table d'anotation qu'on pourra changer au grès de name_test
    
    a='./DonneesPI/timeLapsePhotos_Pi1_0' 
    b='./DonneesPI/timeLapsePhotos_Pi1_1' 
    c='./DonneesPI/timeLapsePhotos_Pi1_2' 
    d='./DonneesPI/timeLapsePhotos_Pi1_3' 
    e='./DonneesPI/timeLAapsePhotos_Pi1_4'

    imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")

    imagettes_PI_0=imagettes[(imagettes["path"]==a) ]


    #imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]=="image_2019-04-30_18-17-14.jpg"]
    imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
    to_drop=['path', 'filename', 'width', 'height', 'classe', 'index']

    im=imagettes1.drop(to_drop,axis=1)
    col=list(im.columns)
    col = col[-1:] + col[:-1]


    annontation_reduit=im[col]
    
    
    
    
    
    
    liste_carre_annote=annontation_reduit[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    
    
    
    #On construit les carrés pour les annotations faites par diff on utilisera proablement une boucle pour les comparer au carré de ref
    #generate_square=table.iloc[:,1:5]
    if filtre_choice=="No_filtre":
        generate_square=table_full.iloc[:,1:5]
    elif filtre_choice=="quantile_filtre":
        generate_square=table_quantile.iloc[:,1:5]
    elif filtre_choice=="RL_filtre":
        generate_square=table_filtre_RL.iloc[:,1:5]
    
    liste_carre_diff=generate_square[['xmin', 'ymin', 'xmax', 'ymax']].apply(to_polygon,axis=1)
    
    #Maintenant on va voir si les carrés des diffs ont suffisament de surfaces en commun avec les carrés des annotations

    
    #Initialisation
    liste_ANNOTATION=[]
    liste_DIFF=[]
    nb_carre_diff=0
    nombre_imagette_oiseau=0    
    
    proportion_limit=0.1
    
    for i in liste_carre_diff:
        max_proportion_test=0
        
        #A chaque fois qu'on change de carre_diff on va remettre le compteur à 0 pour les annotations
        nb_carre_annotation=0
        
        for polygon_ref in liste_carre_annote:
            intersection=polygon_ref.intersection(i)
            proportion=intersection.area/polygon_ref.area
            if proportion>proportion_limit:
                max_proportion_test=proportion
                liste_ANNOTATION.append(nb_carre_annotation)
            
            #On passe au carré un mais peut etre que le compteur devrait se faire plus bas. 
            nb_carre_annotation+=1
        if (max_proportion_test>proportion_limit) :  
            liste_DIFF.append(nb_carre_diff)
            nombre_imagette_oiseau+=1
        
        #On passe au carre_diff suivant 
        nb_carre_diff+=1
    
    
    print( "nombre d'oiseau repérés", nombre_imagette_oiseau)
    
    
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les diffs
        
        
    imageRectangles=imageB.copy()
    #liste_DIFF=[499,610]
    
    for i in liste_DIFF:
        xmin=generate_square["xmin"].iloc[i]
        ymin=generate_square["ymin"].iloc[i]
        xmax=generate_square["xmax"].iloc[i]
        ymax=generate_square["ymax"].iloc[i]
        
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
    
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
    
    
    #En cas de zéro il faut soulever une erreur !!!
    batchImages_test = [batchImages_stack_reshape[i] for i in (liste_DIFF)]
    batchImages=np.vstack(batchImages_test)
    
   
        #Prediction modèle
    model = load_model(neurone_features,compile=False)
    CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
    estimates = CNNmodel.predict(batchImages.reshape(-1,28,28,3))
    
  
    if nombre_imagette_oiseau>0:
        are_there_birds=True
    else:
        are_there_birds=False
    #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
    return are_there_birds




path_image_test="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/image_2019-04-30_18-29-45.jpg"

imageB = cv2.imread(path_image_test)


#name_test="image_2019-04-30_18-27-49.jpg"
name_test="image_2019-04-30_18-29-45.jpg"

birds_is_catched(neurone_features,imageA,imageB,filtre_choice,coef_filtre,path_anotation,name2,name_test="image_2019-05-09_10-19-29.jpg",height=2448,width=3264)







