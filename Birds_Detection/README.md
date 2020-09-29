# Le repertoire regroupe les répertoires nécessaires au fonctionnement de l'interface
# ainsi que la plupart des test d'analyses d'images réalisés par Marc Pozzo




- 
## fonctionnement de l'interface
bin : codes appellés par l'interface pour faire des choses sur le pi + procédures d'évaluations des résultats
code : code de l'interface 

## exploration de pistes
Test_Yolo : scripts d'Entraînement et test avec YoloV2 et YoloV3

Test_VGG scripts de test sur les images

4_Chanels_test : propose un réseau de neurone à 3 canaux de couleurs, plus un canal 
    caractérisant la différence de l'image analysée avec l'image précédante.

find_dominant_colors cherche la couleur de fond majoritaire pour 
    éventuellement créer de nouvelles classes, terres, herbes et arbres 
    (subdivisions de la classe autres pour tenter d'améliorer le modèle)

Imagette_Size : impact des différents types de filtres (ssim, diff basique, ...) 
    sur le nombre et la taille des imagettes extraites 

- Positionnement script pour passer des positions en vue photo en position en vue carte 
- Parameter_GPU premières explorations pour faire fonctionner le GPU

## stockage
- Matériels regroupe les données stockés trop lourdes pour git. Ce fichier correspond à un git caché

