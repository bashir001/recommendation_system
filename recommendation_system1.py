import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")

df.columns
features = ['keywords','cast', 'genres', 'director','original_language']

def combined_features(row):
    return row['keywords']+ "" +row['cast']+ "" +row['genres']+ "" +row['director']+ "" +row['original_language']
    
for feature in features:
    df[feature] = df[feature].fillna("")
df['combined_features']= df.apply(combined_features, axis = 1)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrics = cv.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(count_matrics)

def find_title_from_index(index):
    return df[df.index == index]['title'].values[0]
def find_index_from_title(title):
    return df[df.title == title]['index'].values[0]
    
movie = 'Men in Black II'
movie_index = find_index_from_title(movie)

movie_index

similar_movies = list(enumerate(cosine_sim[movie_index]))

sorted_similar_movie = sorted(similar_movies,key= lambda x:x[1],reverse = True)[1:]

i = 0
for element in sorted_similar_movie:
    print(find_title_from_index(element[0]))
    i=i+1
    if i > 10:
        break