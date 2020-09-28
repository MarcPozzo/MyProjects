#### README #### 


L'objectif est de proposer pour un utilisateur donné un pannel de films conseillés.

Différents modèles ont été entraînés et évalués à partir de deux bases de données décrivant les caractéristiques utilisateurs et des films et donnant la note attribuée à chaque film par ces utilisateurs.

# Pour faire un test immediat
Un modèle de moteur de recommandation  réduit aux films d'actions est opérationnel. Pour le tester voici comment faire :
- Télecharger le répertoire : MovieRecommenderProject
- Ouvrir un terminal et faire un change directory (commande cd), jusqu'à l'intérieur de ce répertoire
- Taper dans le terminal MovieRecommenderEngine/
- Taper python3 Action_Engine.py

Un message vous demandra de vous identifier par un nombre et de prendre ainsi fictivement la place d'un réel utilisateur dont les notations ont été enregistrées dans le passé dans les bases de données.
On vous demandera combien vous souhaiterez qu'on vous conseille de films. Il faudra taper un nombre supérieur ou égal à 1.
On vous demandera ensuite si vous voulez un rappel des films que vous avez déjà consulter, si c'est cela vous intéresse taper Yes.
Enfin les films conseillés s'afficheront.


# Pour reproduire chez vous le travail

Télécharger les 2 datasets : 
- IMDB qui recense 500K films et leurs caractéristiques (année, durée, genre, réal, acteurs...) à télécharger ici : https://www.imdb.com/interfaces/
- MovieLens qui recense 27K films et 138K utilisateurs ayant chacun notés certains de ces films à télécharger ici : https://grouplens.org/datasets/movielens/

- S'assurer de la bonne architecture du dossier dataset, à l'exterieur du git-repo.
- Faire tourner le script MovieRecommenderSystem/tri_data/infos_users pour créer les dataframes qui s'enregistrent dans dataset/data_regression (on en a besoin pour les scripts de regression et de deep-learning).
- Faire tourner le script MovieRecommenderSystem/algorithmes/recommender_system/creation_matrices_similarites jusqu'à l'enregistrement de la matrice movie_similarity.csv 


# Architecture du projet :

../../dataset
- ml-20m
	- genome-scores.csv
	- genome-tags.csv
	- links.csv
	- ratings.csv
	- tags.csv
- data_regression
	- df_movie.csv
	- df_note.csv
	- df_user.csv
- name.basics.tsv
- title.akas.tsv
- title.basics.tsv
- title.crew.tsv
- title.episode.tsv
- title.principals.tsv
- title.ratings.tsv

MovieRecommenderSystem (git-repo)

## Folders
- algorithmes 
- analyses_data 
- new_dataset 
- tri_data 

##Files
- README.md
-Action_Engine.py
- functions_mini_app.py
- JB_corr_a_apporter.ipynb 

Dossier algorithmes

regroupe les différents algorithmes de data-science étudiés ici

- clustering

- théorie des graphes
	- detection_communautes_movies est un fichier utilisant networkX pour tracer des graphes reliant les films entre eux. 2 films sont reliés entre eux si 1 même utilisateur a voté pour les 2. Sont ensuite réalisés des détections de communautés
	- detection_communautes_movies est un fichier reliant les users entre eux via des graphes. 1 arête est créée entre 2 users si ils ont votés pour le même film. Nombre d'arêtes trop élevée, à revoir.

- regression
	- regression_all fait une régression linéaire sur Scikit-Learn en utilisant les informations des films et les informations des utilisateurs, pour prédire la note qu'un utilisateur va donner à un film
- regression_keras fait la même chose, mais sur Keras
- regression_moy est un fichier pourri
- regression_note est un ancien fichier, voir regression_all



- recommender system
	- creation_matrices_similarites permet de créer des matrices modèles train et test sur les notes des utilisateurs. Crée également le fichier movie_similarity, utile pour le filtrage collaboratif.
	- alternating_least_square est un modèle de recommender system 
	- filtrage collaboratif est un autre modèle, basé sur les similarités entre movies, et plus loin dans le fichier sur les similarités entre users (avoir d'abord fait tourner le script creation_matrices_similarites.ipynb
	- reduced... sont des copies des scripts cités plus haut, réalisés sur une portion réduite du dataset.


Dossier new_dataset

regroupe les dataframes créés lors des scripts de MovieRecommenderSystem/tri_data. Lorsque les df sont trop lourds pour être comités, ils sont stockés dans dataset (c'est le cas pour les df créés dans MovieRecommenderSystem/tri_data/infos_users et les matrices créées dans MovieRecommenderSystem/algorithmes/recommendersystem)

- correspondances_Id_movie fait le lien entre les movieId utilisés dans MovieLens, movieId utilisés dans IMDB, et la clé movieId (recréée pour être continue) utilisée par la suite = appelée movieId_ref partout
- imdb_db est le df créé la suite du tri des données IMDB
- movie_df est le df créé à la suite du regroupement des 2 db, donc il a les bons movieId_ref

Dossier tri_data

regroupe les scripts triant les données des 2 datasets, et les regroupant via des clés représentant les films. 

- comp_2_datasets est le script permettant de lier les 2 databases IMDB et MovieLens
- infos_users est un script recoupant des informations sur les utilisateurs (nombre de vote pour chaque utilisateur, moyenne de votes, nombre de vote pour chaque genre de film...)
- tri_IMDB est le script permettant de trier la database IMDB
- tri_MovieLens est le script permettant de trier la base de donnée MovieLens

Plusieurs modèles sont étudiés :
- recommender system
- regression 

A ces modèles de prédiction s'ajoutent des modèles de clustering pour regrouper des utilisateurs et/des movies.
- clustering K-Means
- detection de communautes graphes nx
Les films proposés sont alors issus du recoupement des 2 types de modèles




