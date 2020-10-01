# Le repertoire regroupe les scripts d'évaluation pour les différents classifieurs de deep learning 

Les scripts ci-dessous ont en commun de retourner le nombre de FP de TP et le temps d'execution

## Modèle VGG16
predict_VGG_bis.py moteur VGG avec un série de paramètres testée à la fois

## Modèles Lenet
all_results_Lenet.py : propose le nombre de FP,TP avec plusieurs séries de combinaisons de paramètres testée cf rapport (un seul modèle, déjà entrainé mais évaluation sur tous les dossiers pour toutes les combinaisons de paramètres : ssim ou light, combien on prends en présélection après le random forest, seuil de sélection des oiseaux en fonction des résultats du réseau de neurone. 

Get_FP_TP_FN_Lenet.py: précurseur de all_results_Lenet.py. Détermine le nombre de FP,TP,FN pour le modèle Lenet pour une combinaison de paramètres

## Modèle Yolo V2
my_custom_loss_YoloV2.py : détermine le nombre de FP et TP pour plusieurs paramètres donnés



##Matricre de confusion
Matrice de confusion et courbe FP,TP,FN reproduit à la main pour les modèles VGG16 et Lenet



