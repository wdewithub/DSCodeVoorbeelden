# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:02:58 2018

@author: Wendy
Dit gaat uit dat je ratings van items hebt. Als het enkel binary data zijn zoals clicks, purchases etc: zie
hoe je dit moet aanpassen in de  cursus: sum of similarities !!!!
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import gc

os.chdir(r'''C:\Users\Wendy\DataScience\Recommender systems\Item-based collaborative filtering\Programming assignment\assignment\ii-assignment\data''')

#Reading in the data

movies=pd.read_csv('movies.csv',sep=",",  encoding='latin-1')  #movies and their genre indicators, seperated by pipes
ratings=pd.read_csv('ratings.csv', sep=",", encoding='latin-1') #ratings of user for movies with time-indicator (you could play around with weights ifv recency)
tags=pd.read_csv('tags.csv', sep=",", encoding='latin-1') #tags of user for movies with time indicator


#Compute the mean-adjusted ratings for each movie

movie_mean=ratings.groupby(['movieId'])['rating'].mean()
movie_mean=pd.DataFrame(movie_mean, columns=['rating'])
movie_mean.reset_index(inplace=True)
movie_mean.columns=['movieId', 'avg_rating']

adjratings=pd.merge(ratings, movie_mean, how='left', on='movieId')
adjratings['adj_rating']=adjratings['rating']-adjratings['avg_rating']


#Compute movie_similarities in calculating the cosine similarity between movies mean-adjusted ratings

###First build a mean-adjusted ratings matrix with rows=userId's and columsn=movieId's
adjratings=pd.merge(movies, adjratings, how='left', on='movieId')
user_ratings=adjratings.loc[:,['userId', 'movieId', 'adj_rating']]
user_ratings.set_index(['userId', 'movieId'], inplace=True)

user_ratings_mat = sps.csr_matrix((user_ratings.adj_rating, (user_ratings.index.labels[0], user_ratings.index.labels[1])))

###Bereken similarity matrix van 2 movies

###Bereken product van 2 users hun ratingvectoren
moviecorrelation=np.dot(user_ratings_mat.T, user_ratings_mat)
###Bereken product van normen van ratingsvectoren van 2 movies
norm= user_ratings_mat.toarray()
movienorms=np.linalg.norm(norm, axis=0).reshape(norm.shape[1],1) 
movienormprod=np.dot(movienorms, movienorms.T)
###Cosine similarity is ratingsproduct/product van normen
similaritymatrix=moviecorrelation/movienormprod
###Voeg kolomnamen toe en movieid's als extra kolom
movietitle=movies.title #Dit was al gesorteerd op movieId
movieids=pd.DataFrame(movietitle, columns=['title'])
similaritymatrix_df=pd.DataFrame(similaritymatrix, columns=movieids.title) 
###Only store positive similarities between movies
similaritymatrix_df=similaritymatrix_df.clip(lower=0)
similaritymatrix_df=pd.concat([movieids.reset_index(drop=True), similaritymatrix_df.reset_index(drop=True)], axis=1)


del [[moviecorrelation, norm, movienorms, movienormprod]]
del[similaritymatrix]
gc.collect()

   
#Find the score for each movie for each user. To do so: focus on a user's ratings and use these to calculate
#the weighted average rating and denormalize again

###Transformeer de user_ratings tot een 862x2500 matrix die een 1 geeft als user die movie gerate heeft en 0 anders
###En zet die ook weer in een sparse matrix

user_ratings['fl_rated']=1
user_ratings_flmat = sps.csr_matrix((user_ratings.fl_rated, (user_ratings.index.labels[0], user_ratings.index.labels[1])))


###Bepaal tot slot de score van user u voor item j gebruikmakend van de mean_adjusted ratings van users (user_ratings_mat)
###, de movie similarity matrix (similaritymatrix_df) en de mean_ratings (movie_mean)

simil=sps.csr_matrix(similaritymatrix_df.iloc[:,1:].values)  #Als je deze ook eerst omzet naar sparse matrix, verhoog je performantie van np.dot sterk !!!

moviepred_base=np.dot(user_ratings_mat, simil)

denominator=np.dot(user_ratings_flmat, simil) 

movie_avg=movie_mean.avg_rating.reshape(1,-1)

moviepred=(moviepred_base/denominator)+movie_avg

####Voeg userid's toe als variabele en neem movieid's als kolomnamen
movienames=movies['title']
moviepred=pd.DataFrame(moviepred, columns=movienames) 
userids=ratings.userId.unique() #Dit was al gesorteerd op userId
userids=pd.DataFrame(userids, columns=['userId'])
userids=userids.sort_values(['userId'])
moviepred=pd.concat([userids.reset_index(drop=True),moviepred.reset_index(drop=True)], axis=1)
moviepred=moviepred.set_index(['userId'])
###Dit is nu jouw basis score lijst waarmee je top K movie kandidaten etc kan selecteren per userId 


#Clean up unnecessary memory objects
del[[user_ratings, userids]]
del[[similaritymatrix, similaritymatrix_df]]
del[[movie_avg, movie_mean]]
del[[adjratings]]
gc.collect()


#Als je basket recommendaties wil doen (bijv. obv films die hij deze week gezien heeft of vandaag
# heeft zitten browsen):
#Selecteer in de similarity matrix de rijen van de basketitems en sommeer dan de kolommen
#Neem hier dan de top3 van de gesommeerde similarities

basket=['Jumanji (1995)','101 Dalmatians (1996)']

similaritymatrix_df=similaritymatrix_df.set_index(['title'])
basketsimil=similaritymatrix_df.loc[basket,:]

basketsimil=np.sum(basketsimil, axis=0)
basketsimil=basketsimil.sort_values(ascending=False)
recommend=basketsimil.drop(['Jumanji (1995)','101 Dalmatians (1996)'])[:5]





