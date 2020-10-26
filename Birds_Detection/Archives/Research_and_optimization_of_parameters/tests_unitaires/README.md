*testFunctions.py : fait prédiction sur imagette prédit avec filtre numérique
**image intérieur ou exterieur
** Pour l'instant mauvais résultats

*findsquare.py :on fait une prédiction à partir de fonction sur une zone isolée.
**les imagettes à partir du filtre numériques 
**et les imagettes à partir des annotations d'Alex. 
**Les résultats ne semblent pas cohérent d'où la création de test unitaire.
***Atention les coordonnées des annnotations d'Alex sont bien enregistrées, mais ne sont
mal/pas prises en comptes. 

*find_square_debug.py debug le script ci-dessus.

*reservoir debug
**propose des bouts de code à éventuellement rajouter dans find_square_debug.py comme le zoom des carrés.

*test_unitaire_imagette.py pour une image  exterieur fait une prédiction sur des imagettes pour une gamme de modèle entrainés sur des images avec un plan plus ou moins élargi 
**à partir des coordonnées selon les annotation d'Alex  et des coordonnées pour des images élargies.
**Il faut faire une table recap des résultats
**Objectif faire un point sur les meilleurs résultats sur l'imagette en question (et non
 sur l'ensemble du modèle).

*find_square_debug.py: 
**Incorpore les imagettes dans les predictions
**Il y a peut être une inversion sur les colonnes de table, ymin et xmax qui peuvent provoquer une erreur les prédictions


