#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:19:01 2020

@author: marcpozzo
"""

   
imageRectangles = imageB.copy()
for ic in range(0,len(cnts)):
    

    (x, y, w, h) = cv2.boundingRect(cnts[ic])
    name = (os.path.split(name2)[-1]).split(".")[0]
    name = name + "_" + str(ic) + ".JPG"
    f = pd.Series(dtype= "float64")
    f.xmin, f.xmax, f.ymin, f.ymax = x, x+w, y, y+h
   

        #Maintenant on va ajuster les carrez jusqu'a trouver un resultat positif
    if( (f.xmax-f.xmin)<x_pix_max and (f.ymax-f.ymin)<y_pix_max # birds should less than 500 pixels wide and 350 high
       and (f.xmax-f.xmin)>x_pix_min and (f.ymax-f.ymin)>y_pix_min): # according to distribution in annotations
        subI, o, d, imageRectanglesB = fn.GetSquareSubset(imageB,f,verbose=False)
        subI = fn.RecenterImage(subI,o)
        subI = cv2.resize(subI,(imageSize,imageSize))
        subI = np.expand_dims(subI, axis=0)
        # subI = preprocess_input(subI)
        batchImages.append(subI)
            
        liste_table.append(np.array([[name], [f.xmin], [f.xmax], [f.ymin], [f.ymax]], ndmin = 2).reshape((1,5)))
            
        #table.append(np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5)))
#           table1 = np.array([[name], [x], [x+w], [y], [y+h]], ndmin = 2).reshape((1,5))
    
            #   cv2.rectangle(imageRectangles, (o.xmin,o.ymin), (o.xmax,o.ymax), (255, 0, 0), 2)     
        #écriture des images
#        cv2.imwrite("images_carre/"+h.filename[:-4]+".JPG",img1)
#        
"""




#Deuxième tour


 
"""  
table,index_possible_birds=fn.filtre_quantile(table,coef_filtre,height=2448,width=3264)
#table=table.rename(columns={0: "imagetteName", 1: "xmin", 2 : "xmax", 3 : "ymin" , 4 : "ymax" })
 
#Affecter le max et le min  renommer les variables
liste_xmax=[]
liste_xmin=[]
liste_ymax=[]
liste_ymin=[]
for i in range(len(table)):
    XMAX=table["xmax"].iloc[i]
    XMIN=table["xmin"].iloc[i]
    YMAX=table["ymax"].iloc[i]
    YMIN=table["ymin"].iloc[i] 
        
    #MAX=table["xmax"].iloc[i]=agrandissement(XMIN,XMAX,zoom)
    largeur_min,largeur_max=fn.agrandissement(XMIN,XMAX,zoom)
    largeur_min=int(round(largeur_min))
    largeur_max=int(round(largeur_max))
    liste_xmin.append(largeur_min)
    liste_xmax.append(largeur_max)
        
    profondeur_min,profondeur_max=fn.agrandissement(YMIN,YMAX,zoom)
    profondeur_min=int(round(profondeur_min))
    profondeur_max=int(round(profondeur_max))
    liste_ymin.append(profondeur_min)
    liste_ymax.append(profondeur_max)
        
        
    #On ajoute les éléments de la dataframe provenant des anotations de Alex
liste_xmax=liste_xmax
liste_xmin=liste_xmin
liste_ymax=liste_ymax
liste_ymin=liste_ymin
    
 
    
table["xmax"]=liste_xmax
table["xmin"]=liste_xmin
table["ymax"]=liste_ymax
table["ymin"]=liste_ymin
#table=pd.concat([table,annontation_reduit])

    
batchImages_filtre = [batchImages[i] for i in (index_possible_birds)+list(range(2599,2625)) ]
    
#batchImages_filtre = np.vstack(batchImages_filtre)

#print("features extraction")
"""    
#    preds = filtre_RL.predict(np.array(table.iloc[:,1:], dtype = "float64"))
#features=preprocess_input(np.array(batchImages_filtre))
#features=preprocess_input(np.array(table.iloc[:,1:]))"""



