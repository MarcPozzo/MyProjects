The puropose is to detect a maximum of birds in fields. 
Birds eat seeds in fields and farmers loose a part of their harverst, that is why we want to detect them in order to make them fear. 
Custom Convolutional Neural Network was used such as Lenet or YoloV2.

Below an image of Birds detected (green squares) by Lenet CNN

![FO](https://user-images.githubusercontent.com/30336936/94801959-1fc0f480-03e7-11eb-9986-534e52c07f3a.jpg)


bin : codes appellés par l'interface pour faire des choses sur le pi + procédures d'évaluations des résultats
code : code de l'interface 


Test_Yolo : scripts d'Entraînement et test avec YoloV2 et YoloV3

Model_Results
Run VGG16, Lenet or Yolov2 and displays the number of birds find and the number of False Positiv


4_Chanels_test : propose un réseau de neurone à 3 canaux de couleurs, plus un canal 
    caractérisant la différence de l'image analysée avec l'image précédante.

find_dominant_colors cherche la couleur de fond majoritaire pour 
    éventuellement créer de nouvelles classes, terres, herbes et arbres 
    (subdivisions de la classe autres pour tenter d'améliorer le modèle)

Imagette_Size : impact des différents types de filtres (ssim, diff basique, ...) 
    sur le nombre et la taille des imagettes extraites 

- Positionnement script pour passer des positions en vue photo en position en vue carte 
- Parameter_GPU premières explorations pour faire fonctionner le GPU


- Matériels regroupe les données stockés trop lourdes pour git. Ce fichier correspond à un git caché

Train
Train Different version of custom Lenet neural networks 



