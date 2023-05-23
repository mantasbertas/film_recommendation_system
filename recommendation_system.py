#!/usr/bin/env python
# coding: utf-8

# # Hybrid Film Recommendation System  <a class="tocSkip">
# 
# This project aims to build a **hybrid film recommendation system** that combines multiple approaches for suggesting movies to users. The system will leverage techniques such as *content-based filtering*, *collaborative filtering*, and *popularity-based recommendations* to generate personalized and diverse recommendations. By integrating these methods, we expect to create a more robust and accurate recommendation system that caters to users' preferences and interests while also considering popular and trending movies. The final recommendation will be an aggregation of the individual methods, ensuring a well-rounded and comprehensive set of suggestions for each user.
# 

# In[1]:


import pandas as pd
import numpy as np
import random
import os
import csv
from IPython.display import display, HTML
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder


import datetime

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import wandb
import pickle
from keras.models import load_model

from wandb.keras import WandbCallback
from tensorflow.keras import backend as K

display(HTML("<style>.container { width:90% !important; }</style>"))



#function to accept user watchlist

def get_user_watchlist(file_path):
    watchlist = pd.read_csv(file_path)
    return watchlist




#Simplest, popularity function

def calculate_popularity_score(df, alpha = 0.8, beta=1, gamma=0.002):
    df = df.copy()
    df['norm_numVotes'] = df['numVotes'] / df['numVotes'].max()
    df['norm_averageRating'] = df['averageRating'] / df['averageRating'].max()
    df['norm_startYear'] = (df['startYear'] - df['startYear'].min()) / (df['startYear'].max() - df['startYear'].min())
    df['popularity_score'] = alpha *df['norm_numVotes'] + beta * df['norm_averageRating'] + gamma * df['norm_startYear']
    df = df.drop(columns=['norm_numVotes', 'norm_averageRating', 'norm_startYear'])
    return df


def get_pop_movies_with_random(df, user_ratings_df, num_movies=5, top_percent = 0.01, min_votes= 10000, random_seed=None, genre=None):
    watched_movies = set(user_ratings_df['imdb_id'])
    df = df[~df['tconst'].isin(watched_movies)]
    
    if random_seed is not None:
        random.seed(random_seed)
    df = df[df['numVotes'] >= min_votes]
    if genre is not None:
        df = df[df[genre] == 1]
    top_n = int(df.shape[0] * top_percent)
    top_movies = df.head(top_n)
    num_movies = min(num_movies, top_movies.shape[0])
    selected_movies = top_movies.sample(num_movies, replace=False)
    columns=['primaryTitle', 'tconst', 'startYear', 'averageRating', 'numVotes', 'isAdult', 'runtimeMinutes']
    selected_movies = selected_movies[columns]
    return selected_movies





#Item-based collaborative filtering

def create_user_movie_matrix(ratings_data, min_votes=50, min_ratings=10):
    popular_movies = ratings_data.groupby('imdb_id').filter(lambda x: len(x) >= min_votes)
    active_users = popular_movies.groupby('user_id').filter(lambda x: len(x) >= min_ratings)
    return active_users.pivot_table(index='user_id', columns='imdb_id', values='rating').fillna(0).astype('uint8')

def compute_movie_similarity(user_movie_matrix):
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    return cosine_similarity(user_movie_matrix_filled.T)

def get_top_similar_movies(movie_id, user_movie_matrix, similarity_matrix, n=10):
    movie_idx = user_movie_matrix.columns.get_loc(movie_id)
    similar_movie_indices = np.argsort(similarity_matrix[movie_idx])[::-1][1:n+1]
    similar_movie_ids = [user_movie_matrix.columns[i] for i in similar_movie_indices]
    return similar_movie_ids

def recommend_item_based(user_ratings_df, user_watchlists, movie_data, min_votes=10, num_rec=10):
    user_movie_matrix = create_user_movie_matrix(user_watchlists, min_votes)
    similarity_matrix = compute_movie_similarity(user_movie_matrix)
    watched_movies = set(user_ratings_df['imdb_id'])

    #user_ratings_df = user_ratings_df[user_ratings_df['imdb_id'].isin(user_movie_matrix.columns)]
    total_scores = defaultdict(float)
    
    for _, row in user_ratings_df.iterrows():
        movie_id = row['imdb_id']
        rating = row['rating']
        
        if movie_id not in user_movie_matrix.columns:
            continue
        
        similar_movies = get_top_similar_movies(movie_id, user_movie_matrix, similarity_matrix, num_rec)
        
        for similar_movie in similar_movies:
            if similar_movie in watched_movies:
                continue
            print("Rating: ", rating)
            print("Type of rating: ", type(rating))
            similarity_score = similarity_matrix[
                user_movie_matrix.columns.get_loc(movie_id), user_movie_matrix.columns.get_loc(similar_movie)]
            print("Similarity score: ", similarity_score)
            print("Type of similarity score: ", type(similarity_score))
            total_scores[similar_movie] += similarity_matrix[user_movie_matrix.columns.get_loc(movie_id), user_movie_matrix.columns.get_loc(similar_movie)] * rating
    
    recommended_movies = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)[:num_rec]
    recommended_movie_ids = [movie[0] for movie in recommended_movies]
    
    return movie_data[movie_data['tconst'].isin(recommended_movie_ids)]


# **User-based collaborative filtering**. This approach focuses on finding users who have similar preferences, rather than items that are similar. The idea is that if two users agree on one issue, they are likely to agree on others as well. 



#User-based collaborative filtering

def compute_user_similarity(user_movie_matrix):
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    return cosine_similarity(user_movie_matrix_filled)

def generate_user_based_scores_1(user_movie_matrix, similarity_matrix, user_vector, n=10):
    for movie in user_movie_matrix.columns:
        if movie not in user_vector.index:
            user_vector[movie] = 0  # default rating

    user_vector = user_vector[user_movie_matrix.columns]

    user_similarity = cosine_similarity(user_vector.values.reshape(1, -1), user_movie_matrix)
    top_n_similar_users = user_similarity[0].argsort()[-n:][::-1]

    total_scores = defaultdict(float)

    for similar_user in top_n_similar_users:
        weight = user_similarity[0][similar_user]
        similar_user_ratings = user_movie_matrix.iloc[similar_user]

        for movie_id, movie_rating in similar_user_ratings.iteritems():
            if not np.isnan(movie_rating):
                total_scores[movie_id] += (movie_rating * weight)
    
    return total_scores

def recommend_user_based_collaborative(user_watchlist, user_watchlists, movies_df, num_rec=10):
    user_watchlists = user_watchlists.append(user_watchlist)
    watched_movies = set(user_watchlist['imdb_id'])

    user_movie_matrix = create_user_movie_matrix(user_watchlists)
    similarity_matrix = compute_user_similarity(user_movie_matrix)

    user_vector = user_watchlist.set_index('imdb_id')['rating']
    scores = generate_user_based_scores_1(user_movie_matrix, similarity_matrix, user_vector, num_rec)

    recommended_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_movies = [movie for movie in recommended_movies if movie[0] not in watched_movies][:num_rec]
    recommended_movie_ids = [movie[0] for movie in recommended_movies]

    return movies_df[movies_df['tconst'].isin(recommended_movie_ids)][['primaryTitle', 'tconst', 'startYear', 'averageRating', 'numVotes', 'isAdult', 'runtimeMinutes']]


# Incorporating a **content-based recommendation** system within a movie recommendation engine is key for delivering personalized suggestions tailored to users' unique preferences. This approach complements broader methods like popularity-based and collaborative filtering, ensuring diverse recommendations that reflect individual tastes. By considering specific movie features, content-based recommendations help users discover new, lesser-known titles that align with their interests, enhancing their overall movie-watching experience.



#genre similarity based recommendations
def create_user_profiles(movie_data, user_watchlists):
    genre_columns = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
                     'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                     'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
    
    user_movie_data = user_watchlists.merge(movie_data, left_on='imdb_id', right_on='tconst', how='inner') #merge both df's
    
    user_movie_data['penalty_factor'] = user_movie_data['rating']/ 10
    
    user_movie_data[genre_columns] = user_movie_data[genre_columns].multiply(user_movie_data['rating'] * user_movie_data['penalty_factor'], axis="index")
    
    user_movie_data['weighted_genres'] = user_movie_data[genre_columns].apply(lambda row: np.array(row), axis=1)
    user_profiles = user_movie_data.groupby('user_id')['weighted_genres'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    
    return user_profiles

def recommend_movies_based_on_genre(movie_data, user_watchlists, new_user_watchlist, num_pool=50, num_recommendations=10, alpha=0.7):
    movie_data = movie_data.copy()
    movie_data = movie_data[movie_data['numVotes'] >= 5000]
    movie_data_with_popularity = calculate_popularity_score(movie_data)
    combined_watchlist = pd.concat([user_watchlists, new_user_watchlist])
    user_id = new_user_watchlist.iloc[0, 0]
    
    user_profiles = create_user_profiles(movie_data_with_popularity, combined_watchlist)
    
    user_profile = user_profiles.loc[user_id]
    
    genre_columns = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
                     'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                     'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
    
    genre_matrix  = movie_data_with_popularity[genre_columns].values
    similarity_scores = cosine_similarity([user_profile], genre_matrix)
    
    popularity_scores = movie_data_with_popularity['popularity_score']
    weighted_similarity_scores = alpha * similarity_scores + (1-alpha) * popularity_scores.values.reshape(1, -1)
    
    sorted_movie_indices = np.argsort(weighted_similarity_scores[0])[::-1]
    user_seen_movies = set(new_user_watchlist['imdb_id'])
    
    recommended_movie_ids  = []
    for movie_idx in sorted_movie_indices:
        movie_id = movie_data_with_popularity.iloc[movie_idx]['tconst']
        if movie_id not in user_seen_movies:
            recommended_movie_ids.append(movie_id)
        if len(recommended_movie_ids) >= num_pool:
            break
    
    final_recommendations = random.sample(recommended_movie_ids, num_recommendations)
    
    columns=['primaryTitle', 'tconst', 'startYear', 'averageRating', 'numVotes', 'isAdult', 'runtimeMinutes']
    return movie_data_with_popularity.loc[movie_data_with_popularity['tconst'].isin(final_recommendations)][columns]



# add user_based and item_based scores to the recommendations

def compute_user_vector(user_watchlist, user_movie_matrix):
    user_vector = pd.Series(0, index=user_movie_matrix.columns)
    for _, row in user_watchlist.iterrows():
        movie_id = row['imdb_id']
        if movie_id in user_vector.index:
            user_vector[movie_id] = 1
    return user_vector

def get_top_similar_movies_item_based(movie_id, user_movie_matrix, similarity_matrix, n=10):
    if movie_id in user_movie_matrix.columns:
        movie_idx = user_movie_matrix.columns.get_loc(movie_id)
        similar_movie_indices = np.argsort(similarity_matrix[movie_idx])[::-1][1:n+1]
        similar_movie_ids = [user_movie_matrix.columns[i] for i in similar_movie_indices]
        similar_movie_scores = similarity_matrix[movie_idx, similar_movie_indices]
        return dict(zip(similar_movie_ids, similar_movie_scores))
    else:
        return {}


    
def get_top_similar_users(user_id, user_movie_matrix, similarity_matrix, n=10):
    if user_id in user_movie_matrix.index:
        user_idx = user_movie_matrix.index.get_loc(user_id)
        similar_user_indices = np.argsort(similarity_matrix[user_idx])[::-1][1:n+1]
        similar_user_ids = [user_movie_matrix.index[i] for i in similar_user_indices]
        return similar_user_ids
    else:
        return []


def generate_item_based_scores(user_vector, user_movie_matrix, similarity_matrix, n=10):
    scores = {}
    for movie_id in user_vector[user_vector > 0].index:
        movie_scores = get_top_similar_movies_item_based(movie_id, user_movie_matrix, similarity_matrix, n)
        scores.update(movie_scores)
    return scores

def generate_user_based_scores(user_vector, user_movie_matrix, similarity_matrix, n=10):
    similarity_scores = cosine_similarity(user_vector.values.reshape(1, -1), user_movie_matrix)[0]
    top_similar_users = np.argsort(similarity_scores)[::-1][:n]
    scores = user_movie_matrix.iloc[top_similar_users].mean()
    return scores




#assign user to one of 5 clusters (5 choosen with the elbow method)

def get_user_cluster(user_vector, user_movie_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(user_movie_matrix)
    user_cluster = kmeans.predict(user_vector.values.reshape(1, -1))[0]
    return user_cluster





#recommendation function, takes recommendations from different systems and joins them into 1, calculates additional features and outputs data for inference

def recommend_movies(user_watchlist, user_watchlists_path, movie_database_path, num_recommendations=10, random_seed=None):
    columns = ['primaryTitle', 'tconst', 'startYear', 'averageRating', 'numVotes', 'isAdult', 'runtimeMinutes']
    movies_df = pd.read_csv(movie_database_path)
    user_watchlists = pd.read_csv(user_watchlists_path)


    user_watchlists = user_watchlists.append(user_watchlist)
    movies_df = movies_df[movies_df['numVotes'] >= 5000]  # filtering out movies with less than 5000 votes

    movies_df = calculate_popularity_score(movies_df)
    
    # popular movies
    popular_movies = get_pop_movies_with_random(movies_df, user_watchlist, num_movies=num_recommendations, random_seed=random_seed)
    print('step 1 done')
    
    # item based colaborative filtering
    collaborative_movies = recommend_item_based(user_watchlist, user_watchlists, movies_df, num_rec=num_recommendations)
    collaborative_movies = collaborative_movies[columns]
    print('step 2 done')
    
    # user based colaborative filtering
    user_movie_matrix = create_user_movie_matrix(user_watchlists)
    user_based_movies = recommend_user_based_collaborative(user_watchlist, user_watchlists, movies_df, num_rec=num_recommendations)
    user_based_movies = user_based_movies[columns]
    print('step 3 done')
    
    # genre based recommendations
    genre_based_movies = recommend_movies_based_on_genre(movies_df, user_watchlists, user_watchlist, num_recommendations=num_recommendations)
    genre_based_movies = genre_based_movies[columns]
    print('step 4 done')
    
    # combining all recommendations
    recommended_movies = pd.concat([popular_movies, collaborative_movies, user_based_movies, genre_based_movies])
    
    

    # compute the similarity matrices
    item_similarity_matrix = compute_movie_similarity(user_movie_matrix)
    user_similarity_matrix = compute_user_similarity(user_movie_matrix)
    
    # compute the collaborative filtering scores
    user_vector = compute_user_vector(user_watchlist, user_movie_matrix)
    item_based_scores = generate_item_based_scores(user_vector, user_movie_matrix, item_similarity_matrix)
    user_based_scores = generate_user_based_scores(user_vector, user_movie_matrix, user_similarity_matrix)
    print('step 5 done')
    
    # add the scores to the recommended_movies DataFrame
    recommended_movies['collab_item_filter_score'] = recommended_movies['tconst'].apply(lambda x: item_based_scores.get(x, 0))
    recommended_movies['collab_user_filter_score'] = recommended_movies['tconst'].apply(lambda x: user_based_scores.get(x, 0))
    
    # cluster user
    user_cluster = get_user_cluster(user_vector, user_movie_matrix, num_clusters=5) # clusterizing using 5 clusters
    recommended_movies['user_cluster'] = user_cluster
    
    #change release year to year since release
    current_year = datetime.datetime.now().year
    recommended_movies['years_since_release'] = current_year - recommended_movies['startYear']

    return recommended_movies


# **The inference step** in my recommendation system involves applying the trained model on the feature-set obtained from the movie recommendation phase. It generates a predicted user rating for each recommended movie, which is then used to rank these movies.


# inference step

def prepare_for_inference(recommendations, movie_database_path, scaler_path):
    genre_columns = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History',
                 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
                 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western', '\\N']

    scaler = pickle.load(open(scaler_path, 'rb'))
    movies_df = pd.read_csv(movie_database_path)
    genre_df = movies_df[['tconst'] + genre_columns]
    recommendations = pd.merge(recommendations, genre_df, on='tconst', how='left')
    
    recommendations = pd.concat([
        recommendations.drop('user_cluster', axis=1),
        pd.get_dummies(recommendations['user_cluster'], prefix='user_cluster')
    ], axis=1)

    for i in range(5):
        if f'user_cluster_{i}' not in recommendations.columns:
            recommendations[f'user_cluster_{i}'] = 0

    scale_cols = ['years_since_release', 'averageRating', 'numVotes', 'runtimeMinutes', 'collab_user_filter_score', 'collab_item_filter_score']
    no_scale_cols = ['isAdult', 'tconst'] + genre_columns + [f'user_cluster_{i}' for i in range(5)]

    recommendations[scale_cols] = scaler.transform(recommendations[scale_cols])
    
    final_data = pd.concat([recommendations[scale_cols], recommendations[no_scale_cols]], axis=1)

    return final_data


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 


def inference(prepared_data, model_path):
    model = load_model(model_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    movie_ids = prepared_data['tconst'].values # save ids for later

    prepared_data = prepared_data.drop(columns=['tconst'])

    predictions = model.predict(prepared_data)

    results = list(zip(movie_ids, predictions.flatten()))
    results.sort(key=lambda x: x[1], reverse=True)

    return results





# complete pipeline

def movie_recommendation_pipeline(user_watchlist, user_watchlists_path, movie_database_path,
                                  scaler_path, model_path, num_recommendations=10, random_seed=None):
    
    recommendations = recommend_movies(user_watchlist, user_watchlists_path,
                                       movie_database_path, num_recommendations, random_seed)
    print('Recommendation step complete, preparing for inference')
    
    recommendations = prepare_for_inference(recommendations, movie_database_path, scaler_path)

    predictions = inference(recommendations, model_path)
    
    results_df = recommendations.copy()
    results_df['predicted_rating'] = [prediction[1] for prediction in predictions]
    results_df['imdb_id'] = [prediction[0] for prediction in predictions]
    
    required_cols = ['imdb_id', 'years_since_release', 'averageRating', 'numVotes', 
                     'isAdult', 'runtimeMinutes', 'collab_user_filter_score', 
                     'collab_item_filter_score', 'predicted_rating']
    
    results_df = results_df[required_cols]
    results_df.sort_values(by='predicted_rating', ascending=False, inplace=True)


    return results_df.head(10)


# In[44]:


### data needed to run the function: user ratings (imdb export), many users ratings (scraped data), film data (imdb export)



# results = movie_recommendation_pipeline('data/user_ratings_data/test_ratings.csv', 'data/user_ratings_data/user_watchlists.csv', 'data/film_data/prepared_film_data.csv',
#                                         'models/scaler.pkl', 'models/properly_scaled_model.h5')




