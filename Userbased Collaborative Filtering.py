# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:02:58 2018

@author: Wendy
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import gc

os.chdir(r'''C:\\Users\Wendy\DataScience\Recommender systems\UU collaborative filtering\Programming assignment\uu_assignment\data''')

#Reading in the data

movies=pd.read_csv('movies.csv',sep=",",  encoding='latin-1')  #movies and their genre indicators, seperated by pipes
ratings=pd.read_csv('ratings.csv', sep=",", encoding='latin-1') #ratings of user for movies with time-indicator (you could play around with weights ifv recency)
tags=pd.read_csv('tags.csv', sep=",", encoding='latin-1') #tags of user for movies with time indicator


#Compute the mean-adjusted ratings for each user

user_mean=ratings.groupby(['userId'])['rating'].mean()
user_mean=pd.DataFrame(user_mean, columns=['rating'])
user_mean.reset_index(inplace=True)
user_mean.columns=['userId', 'avg_rating']

adjratings=pd.merge(ratings, user_mean, how='left', on='userId')
adjratings['adj_rating']=adjratings['rating']-adjratings['avg_rating']


#Compute user_similarities in calculating the cosine similarity between users mean-adjusted ratings

###First build a mean-adjusted ratings matrix with rows=userId's and columsn=movieId's
adjratings=pd.merge(movies, adjratings, how='left', on='movieId')
user_ratings=adjratings.loc[:,['userId', 'movieId', 'adj_rating']]
user_ratings.set_index(['userId', 'movieId'], inplace=True)

user_ratings_mat = sps.csr_matrix((user_ratings.adj_rating, (user_ratings.index.labels[0], user_ratings.index.labels[1])))


#Dit is een veel tragere code dan als je met np.dot etc gaat werken: 
####Next calculate cosine similarity between all users based on this matrix (For large databases you can use
##numpy functionalities to avoid for loops: np.dot(user_mat, user_mat.T) en dan delen door product van euclidean norms)
#
#num_users=user_ratings_mat.shape[0]
#similaritymatrix=np.zeros((num_users, num_users))
#for user in range(num_users):
#    print (user)
#    for comp_user in range(num_users):
#        similaritymatrix[user, comp_user]=cosine_similarity(user_ratings_mat[user], user_ratings_mat[comp_user])
#
##Haal de userid's op om als kolomnamen en rijnamen toe te voegen aan de similaritymatrix
#
#userids=ratings.userId.unique() #Dit was al gesorteerd op userId
#userids=pd.DataFrame(userids, columns=['userId'])
#userids=userids.sort_values(['userId'])
#similaritymatrix=pd.DataFrame(similaritymatrix, columns=userids.userId) 
#similaritymatrix=pd.concat([userids.reset_index(drop=True), similaritymatrix.reset_index(drop=True)], axis=1)



#Check whether the below leads to the same results as the similarity matrix

###Bereken product van 2 users hun ratingvectoren
usercorrelation=np.dot(user_ratings_mat, user_ratings_mat.T)
###Bereken product van normen van ratingsvectoren van 2 users
norm= user_ratings_mat.toarray()
usernorms=np.linalg.norm(norm, axis=1).reshape(norm.shape[0],1) 
usernormprod=np.dot(usernorms, usernorms.T)
###Cosine similarity is ratingsproduct/product van normen
similaritymatrix=usercorrelation/usernormprod
###Voeg kolomnamen toe en userid's als extra kolom
userids=ratings.userId.unique() #Dit was al gesorteerd op userId
userids=pd.DataFrame(userids, columns=['userId'])
userids=userids.sort_values(['userId'])
similaritymatrix_df=pd.DataFrame(similaritymatrix, columns=userids.userId) 
similaritymatrix_df=pd.concat([userids.reset_index(drop=True), similaritymatrix_df.reset_index(drop=True)], axis=1)
####Maak van userId de rij index
#similaritymatrix_df=similaritymatrix_df.set_index(['userId'])

del [[usercorrelation, norm, usernorms, usernormprod]]
gc.collect()

   
#For each item find the 30 nearest neighbours among the users who rated the item and who have a positive
#similarity to the user and score the item. Refuse to score the item if you have 2 users or less.

###Bepaal eerst per userId de top30NN (voorwaarde: minstens 2 neighbours met een positive correlatie)

def top30_neighbours(user):
    print(user)
    user_simil=similaritymatrix_df.loc[similaritymatrix_df.userId==user,:].iloc[:,1:]
    user_simil_sorted=user_simil.squeeze().sort_values(ascending=False)
    user_simil_sorted=user_simil_sorted.loc[(user_simil_sorted > 0) & (user_simil_sorted < 1)] #Enkel positief gecorreleerde en niet zichzelf
    if len(user_simil_sorted)>1:
        top30_NN=user_simil_sorted.index.values[:30]
    else:
        top30_NN=[]
    top30_dict[user]=top30_NN
    return top30_dict

top30_dict={}
similaritymatrix_df.userId.apply(top30_neighbours)

###Maak van de top30_dictionary terug een sparse matrix

top30_df=pd.DataFrame(top30_dict).T.stack()
top30_df=pd.DataFrame(top30_df)
top30_df.columns=['NNid']
top30_df['waarde']=1
top30_df.reset_index(inplace=True)
top30_df.columns=['userId', 'rang', 'NNid', 'waarde']
top30_df.drop('rang', axis=1, inplace=True)
top30_df=top30_df.iloc[:-1,:]
top30_df.set_index(['userId', 'NNid'], inplace=True)
#top30_df.columns=['userId', 'NN_id', 'similarity']
#top30_df.set_index(['userId','NN_id'], inplace=True)

top30_matind = sps.csr_matrix((top30_df.waarde, ((top30_df.index.labels[0], top30_df.index.labels[1]))))
top30_mat=top30_matind.multiply(similaritymatrix)

#top30_mat.todense()[1]
#print(top30_mat[0])


###Transformeer de user_ratings tot een 862x2500 matrix die een 1 geeft als user die movie gerate heeft en 0 anders
###En zet die ook weer in een sparse matrix

user_ratings['fl_rated']=1
user_ratings_flmat = sps.csr_matrix((user_ratings.fl_rated, (user_ratings.index.labels[0], user_ratings.index.labels[1])))


###Bepaal tot slot de score van user u voor item j gebruikmakend van top30NN similarity (top30_mat)
###en de mean_adjusted ratings van users (user_ratings_mat) en de mean_ratings (user_mean)

moviepred_base=np.dot(top30_mat, user_ratings_mat)

denominator=np.dot(top30_mat, user_ratings_flmat) 

user_avg=user_mean.avg_rating.reshape(-1,1)

moviepred=(moviepred_base/denominator)+user_avg

####Voeg userid's toe als variabele en neem movieid's als kolomnamen
movienames=movies['title']
moviepred=pd.DataFrame(moviepred, columns=movienames) 
moviepred=pd.concat([userids.reset_index(drop=True),moviepred.reset_index(drop=True)], axis=1)
moviepred=moviepred.set_index(['userId'])
###Dit is nu jouw basis score lijst waarmee je top K movie kandidaten etc kan selecteren per userId 
###movies met NAN value zijn door de 30 NN van die user niet gerate en kan je dus niets voor geven
###Bijv. Gone Girl (1994) voor userid 320


del[[user_ratings, user_ratings1, user_simil, top30_df, top30_dict]]
del[[similaritymatrix, similaritymatrix_df]]
del[[user_avg, user_mean, user_simil_sorted]]
del[[adjratings]]
gc.collect()









