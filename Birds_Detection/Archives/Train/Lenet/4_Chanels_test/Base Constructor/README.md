#### README #### 


#Générer 4ème canal, entraîner et évaluer sur imagettes



##Fichier Get_fp_4C.py##
génère des listes avec les fp pour chaque images tests au format 28,28,4. Il ne resteras plus qu'a concaténer et à les transformer sous 	forme d'array (-1,28,28,4), fichier précurseur 


##Fichier Get_GBR.ipynb##
Génère les batchs des imagettes en 28,28,4 dont un 4ème canal correspondant à la différence de gris 
	entre l'image étudiée et la précédante. La transformation en grise se fait à partir de la couleur HSV. Les batchs sont stockés 		dans /mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/4D_Pictures/Alex_db/

##Fichier Get_HSV.ipynb##
Génère les batchs des imagettes en 28,28,4 dont un 4ème canal correspondant à la différence de gris entre l'image étudiée et la 		précédante. La transformation en grise se fait à partir de la base HSV. Ce qui a l'air de mieux marcher. Les batchs sont 	stockés dans /mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/4D_Pictures/Alex_db/



#Test sur les images

##4C_test_unitaire.py##

Test résultat directement sur les images entières et s'appuie sur les modèles fait dans la partie modèle 	et enregistrés dans :
	/mnt/BigFast/VegaFastExtension/Rpackages/c3po_all/c3po_interface_mark/Materiels/Models/	4Chanels_6Classes



Un file est utilisé pour stocker les batchs fp sous forme de liste
	/mnt/VegaSlowDataDisk/c3po_interface_mark/Materiels/4D_Pictures/FP/Neurone_name/dossier0/






