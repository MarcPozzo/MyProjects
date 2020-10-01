#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:09:54 2020

@author: marcpozzo
"""

#!/usr/bin/env python3
# 
# -*- coding: utf-8 -*-

#L'objectif de ce script est de comptabiliser les faux négatifs
#Il va surement falloir une autre fonction




#Un faux négatif c'est un élement classé oiseau mais qui en réalité est de classe autre.
#Peu de faux négatif en revanche, la regression quantile n'a pas l'air de bien fonctionné. 

#Importation packages

from os import chdir
chdir("/mnt/VegaSlowDataDisk/c3po_interface/bin")
import pandas as pd
import functions as fn
from keras.models import Model, load_model

#Paramètres à choisir

name_ref="image_2019-04-30_17-47-10.jpg"
coverage_threshold=0.8

#Pour filtre quantile







path_images="/mnt/VegaSlowDataDisk/c3po/Images_aquises/DonneesPI/timeLapsePhotos_Pi1_0/"
imagettes=pd.read_csv("/mnt/VegaSlowDataDisk/c3po/Images_aquises/imagettes.csv")
imagettes_PI_0=imagettes[(imagettes["path"]=="./DonneesPI/timeLapsePhotos_Pi1_0") ]

#Les seules imagettes qui nous intéressent  sont celles des oiseaux pas celle de la terrer
imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]









#Le code fait des répitions pour rien s'il y a plusieurs imagettes
fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold)
dict_images_catched[name_test]=catched_bird
    
len(dict_images_catched)





image_reussie=0
liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test:
    print(name_test)
    if dict_images_catched[name_test]==len(imagettes_PI_0[imagettes_PI_0["filename"]==name_test]):
        image_reussie+=1


#Attention ici ça fonctionne car on a enlevé ground

nb_imagettes_oiseaux=len(imagettes_PI_0) 
nb_catched_imagettes=sum(dict_images_catched.values())
nb_images_oiseaux=len(liste_name_test)


print("le nombre d'images sur lesquelles toutes les imagettes sont identifiées est",image_reussie)
print("le pourcentage d'image sur laquelle l'extraction se fait coorectement est : ",image_reussie/nb_images_oiseaux )
print("le pourcentage d'imagettes extraites parmi les imagettes d'oiseau est de : ",nb_catched_imagettes/nb_imagettes_oiseaux)





liste_name_test=list(imagettes_PI_0["filename"].unique())
for name_test in liste_name_test[0]:
    print(name_test)
    estimates,nb_birds=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold)
    
  
coverage_threshold=0.1
#name_test=liste_name_test[0]
name_test="image_2019-04-30_18-25-35.jpg"
estimates,nb_birds=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,limit_area_square=70)
estimates,nb_birds=fn.birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,limit_area_square=70000000000)
estimates



imagettes_PI_0[imagettes_PI_0["filename"]==name_test]

liste_prediction=list(estimates.argmax(axis=1))
pourcentage_birds_predict=(len(liste_prediction)-liste_prediction.count(0))/len(liste_prediction)






coverage_threshold=0.5
contrast=-5
neurone_features="/mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/iteration_models/z1.2_r_20_drpt_0.1"
blockSize=11
nombre_imagette_oiseau_match,pourcentage_birds_predict,nb_oiseaux_a_reperer,birds_predict=birds_is_catche_boucle(path_images,name_test,name_ref,coverage_threshold,10000000000000,contrast,neurone_features,blockSize)



























































print(name_ref,name_test)
print(coverage_threshold,neurone_features)
print(blockSize)


"""image_2019-04-30_18-16-57.jpg image_2019-04-30_18-17-14.jpg
0.5 /mnt/VegaSlowDataDisk/c3po/Chaine_de_traitement/Train_imagettes_annotées/type_oiseau/pre_trained_models/zoom_models/6c_rob
19
53 faux positif sur 668
soir moins de 8%
prédit pigeon au lieu de Corneille
"""




def faux_positifs(path_images,name_test,name_ref,coverage_threshold,limit_area_square,contrast,neurone_features,blockSize,filtre_choice,coef_filtre=coef_filtre,height=2448,width=3264):

    
    image_ref=path_images+name_ref
    image_test=path_images+name_test
    imageA=cv2.imread(image_ref)
    imageB=cv2.imread(image_test)
    
    #On va comparer les images de ici et probablement test unitaires
    batchImages = []
    liste_table = []
    imageSize= 28
    cnts=fn.filtre_light(imageA,imageB,blockSize=blockSize,contrast=contrast)
    #On récupère les coordonnées des pixels différent par différence
    for ic in range(0,len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[ic])
        name = (os.path.split(name_test)[-1]).split(".")[0]
        name = name + "_" + str(ic) + ".JPG"
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
    """
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
    #Attention ça ne fonction qu'avec le PI_0 pour les autres dossiers il faut rajouter d'autres classes 
    imagettes_PI_0=imagettes_PI_0[imagettes_PI_0["classe"]!="ground"]

    #imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]=="image_2019-04-30_18-17-14.jpg"]
    imagettes1=imagettes_PI_0[imagettes_PI_0["filename"]==name_test]
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
    liste_DIFF_match=[]
    #nb_carre_diff=0
    nombre_imagette_oiseau_match=0    

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
            liste_DIFF_match.append(position_maximum)
            nombre_imagette_oiseau_match+=1
            #Au lieu de faire des +1 comme ça on va plutot lui demander la postion du max 
        
        
        
        
        
    nb_oiseaux_a_reperer=len(liste_carre_annote)
    print( "nombre d'oiseau repérés", nombre_imagette_oiseau_match)
    
    
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les diffs
        
     
    imageRectangles=imageB.copy()
    #liste_DIFF=[499,610]
    
    for i in liste_DIFF_match:
        xmin=generate_square["xmin"].iloc[i]
        ymin=generate_square["ymin"].iloc[i]
        xmax=generate_square["xmax"].iloc[i]
        ymax=generate_square["ymax"].iloc[i]
        
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (255, 0,0), 2) 
    
    #On va initialiser xmin,ymin .... de sorte de n'avoir à changer que la valeur de i pour les annontations
    """
    #liste_ANNOTATION=[4,8]
    for i in liste_ANNOTATION:
        xmin=annontation_reduit["xmin"].iloc[i]
        ymin=annontation_reduit["ymin"].iloc[i]
        xmax=annontation_reduit["xmax"].iloc[i]
        ymax=annontation_reduit["ymax"].iloc[i]
    
        cv2.rectangle(imageRectangles, (xmin,ymin), (xmax,ymax), (0, 255,0), 2) 
    #On va essayer d'écrites maintenant les carrés qui sont censés repérés les oiseaux et vérifier qu'ils matchent bien avec des oiseaux ( pas de décalage d'indices par exemple)
    """
    
    cv2.imwrite("Output_images/annotation_is_match.jpg",imageRectangles)
    
    
    #En cas de zéro il faut soulever une erreur !!! #c'est bien un array dim  (669, 28, 28, 3)
    
    liste_batch_images=list(range(len(batchImages_stack_reshape)))
    liste_DIFF_not_match=set(liste_batch_images)-set(liste_DIFF_match)
    batchImages_imagettes_match = [batchImages_stack_reshape[i] for i in (liste_DIFF_match)]
    batchImages_imagettes_not_match=[batchImages_stack_reshape[i] for i in (liste_DIFF_not_match)]
    #
    try:
        batchImages_match=np.vstack(batchImages_imagettes_match)
        batchImages_not_match=np.vstack(batchImages_imagettes_not_match)
       
            #Prediction modèle
        model = load_model(neurone_features,compile=False)
        CNNmodel = Model(inputs=model.input, outputs=model.layers[-1].output)
        estimates_annotation_matched = CNNmodel.predict(batchImages_match.reshape(-1,28,28,3))
        estimates_annotation_not_matched = CNNmodel.predict(batchImages_not_match.reshape(-1,28,28,3))
        
    
        liste_prediction_match=list(estimates_annotation_matched.argmax(axis=1))
        birds_predict=(len(liste_prediction_match)-liste_prediction_match.count(0))
        pourcentage_birds_predict=birds_predict/len(liste_prediction_match)
        
        #D'abord on va transformer estimates avec préféré le max
        #Dans le tableau il faudra convertir la classe string en classe nombre
        #Puis ensuite on se demande si les deux sont égaux
        liste_prediction_not_match=list(estimates_annotation_not_matched.argmax(axis=1))
        fp=(len(liste_prediction_not_match)-liste_prediction_not_match.count(0))   
        pourcentage_birds_mistake_predict=(faux_positif)/len(liste_prediction_not_match)
             
        
        
        
        #Il faut s'assurer que estimates est seulement pour les imagettes tout d'abord (puis ensuite on verra)
      
        if nombre_imagette_oiseau_match>0:
            are_there_birds=True
        else:
            are_there_birds=False
        #On va essayer maintenant de faire des prédictions  on l'intègrera ensuite dans les carrés
    except ValueError:
        print("il n'a pas d'imagettes trouvées !")
        pourcentage_birds_predict=0
        birds_predict=0
    return fp,birds_predict


name_ref="image_2019-04-30_18-16-57.jpg"   
name_test="image_2019-04-30_18-17-14.jpg"  #shape (720, 1280, 3)



name_test="image_2019-04-30_18-40-19.jpg"
name_ref="image_2019-04-30_18-40-36.jpg"



coef_filtre=pd.read_csv("/mnt/VegaSlowDataDisk/c3po_interface/bin/testingInputs/coefs_filtre_RQ.csv")

fp,birds_predict=faux_positifs(path_images,name_test,name_ref,coverage_threshold,limit_area_square,contrast,neurone_features,blockSize,"quantile_filtre")
print("faux positifs",fp)
print("birds_predict",birds_predict)

#quantile_filtre
#No_filtre
