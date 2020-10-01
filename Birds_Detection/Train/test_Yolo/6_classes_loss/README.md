# Tutoriel tensorflow
## Réseau Yolo


-train_ter.py est un script d'entrainement pour Yolo V2
-train_with_progression.py est un script d'entrainement pour Yolo V2 qui affiche l'évolution de la fonction de perte
-inference.py est un script à lancer depuis le terminal (python3 inference.py) permet d'obtenir les résustats de manière interractives directement sur les photos
- my_custom_loss.py synthétise les résultats sur l'échantillon de test ou d'entraînement.
- map.py synthétise les résultats avec F1 score et précision. Ce script est moins personnalisé pour notre étude de cas.
- config.py contient certains paramètres liés au nombre de classe ou bien aux bouding boxes (nombre, taille).
- model.py script du modèle Yolo V2 utilisé avec Resnet
- find_box_size.py essaie de déterminer des tailles optimales pour les bouding_box
- common_2.py est un script de fonctions supports pour les autres script
- split_db.py permet de séparer la base de données en échantillon de test et d'entraînement
-images.py propose des trans

Les scripts écrits dans ce répertoire s'inspirent fortement de la vidéo disponible à l'adresse suivante: https://www.youtube.com/watch?v=oQ0436IJUWc


