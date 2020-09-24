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

#changer the next movie you will love si sing ...

#Libraries
import pandas as pd
import numpy as np
import os
from os import chdir
chdir("/Users/marcpozzo/Documents/Projet_Git/Projet_Git/MovieRecommenderProject/MovieRecommenderEngine/algorithmes/Movie_Engine")
import function_good_app as fn

#Tables
ratings=pd.read_csv('../../new_dataset/action_movies.csv',index_col=0)
ratings.drop(["userId"],inplace=True,axis=1)
movies=pd.read_csv('../../../../../dataset/ml-20m/movies.csv')
df_ref=pd.read_csv('../../new_dataset/correspondances_Id_movie')


#Parameters to select
user = input("Type a number between 0 and 1168 to indicate your IdUser : ")
nb_movies_asked= input("Type the number of moovies you want to watch : ")
user=int(user)
nb_movies_asked=int(nb_movies_asked)




#Get movie_seen and not seen
action_movies_list=list(ratings["movie_Action"][ratings["Id_Action"]==user].unique())
movies_seen_by_user = ratings[ratings["movie_Action"].isin(action_movies_list)]
movies_seen=fn.get_movie_seen(movies,df_ref,action_movies_list,ratings)
matrice_movie_no_seen,movie_no_seen_new_ref=fn.get_matrice_new(ratings,movies_seen)
print("The list of movies I advice you is below")
print("this is the list of movies already seen : ",list(movies_seen["title"].unique()))


#Get movies adviced
user_similarity=fn.get_user_similarity(movies_seen_by_user)
movies_rates_prediction=np.argsort(fn.pred_user(matrice_movie_no_seen, user_similarity, user))[-1:-1-nb_movies_asked:-1]
movies_adviced=fn.get_movie_advice(movie_no_seen_new_ref,df_ref,movies,movies_rates_prediction)


