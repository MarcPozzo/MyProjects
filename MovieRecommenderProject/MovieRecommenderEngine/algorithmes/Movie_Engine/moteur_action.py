#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:17:35 2020

@author: utilisateur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:42:18 2020

@author: utilisateur
"""

#Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as dist
from tqdm import tqdm
import seaborn as sns
chdir("/Users/marcpozzo/Documents/Projet_Git/Projet_Git/MovieRecommenderProject/MovieRecommenderEngine/algorithmes/Movie_Engine")
import function_good_app as fn

#Tables
ratings=pd.read_csv('../../new_dataset/action_movies.csv',index_col=0)
movies=pd.read_csv('../../../../../dataset/ml-20m/movies.csv')
df_ref=pd.read_csv('../../new_dataset/correspondances_Id_movie')


#Parameters to select
user = input("Type a number between 0 and 1168 to indicate your IdUser : ")
nb_movies_asked= input("Type the number of moovies you want to watch : ")
user=int(user)
nb_movies_asked=int(nb_movies_asked)




n_users = ratings.Id_Action.unique().shape[0]
n_movies = ratings.movie_Action.unique().shape[0]
ratings_action=ratings["movie_Action"]

action_movies_list=list(ratings_action[ratings["Id_Action"]==user].unique())
films_vus_par_u = ratings[ratings_action.isin(action_movies_list)]
movies_seen_by_user = ratings[ratings_action.isin(action_movies_list)]
#In a first we suppose assume that there is no user to remove but only movies to reindex
n_users=len(movies_seen_by_user["Id_Action"].unique())
n_movies=len(movies_seen_by_user["movie_Action"].unique())

movies_seen=fn.get_movie_seen(movies,df_ref,action_movies_list,ratings)
print("Bravo, you have already seen ",str(n_movies) + " movies")
print("The list of movies I advice you is below")
print("this is the list of movies already seen : ",list(movies_seen["title"].unique()))

user_similarity=fn.get_user_similarity(movies_seen_by_user,n_movies,n_users)

matrice_new,movie_no_seen_new_ref=fn.get_matrice_new(ratings,movies_seen)

movies_adviced=fn.get_movie_advice(nb_movies_asked,matrice_new,movie_no_seen_new_ref,ratings,df_ref,movies,user_similarity,user)

