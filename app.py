#!/usr/bin/env python
# coding: utf-8

# ## I create content base recommendation system for movies
# 
# 

# ### Import our necessary library

# In[1]:


import numpy as np
import pandas as pd
import json
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# ###  Load our data

# In[65]:


movies_data=pd.read_csv('m.csv')
movies_data.head(3)


# ### Select our features

# In[3]:


selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# ### Preprocessing data

# In[4]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[5]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# ### Convert text to numerical vector

# In[6]:


vectorizer = TfidfVectorizer()


# In[7]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[8]:


print(feature_vectors )


# ### Calculate cosine_similarity between all vectors

# In[9]:


similarity = cosine_similarity(feature_vectors)


# In[10]:


similarity = cosine_similarity(feature_vectors)


# ### Run a function to recommend movies
# ####  The get_close_matches() function returns a list of close matched strings that satisfy the cutoff.

# In[44]:


list_of_all_titles = movies_data['title'].tolist()

def get_recommendations(movie_name):
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    sim_scores = list(enumerate(similarity[index_of_the_movie]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:70]
    movie_indices = [i[0] for i in sim_scores]
    a = set(movies_data[['title']].iloc[movie_indices]['title'])
    return a

def get_recommendations2(movie_name):
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    sim_scores = list(enumerate(similarity[index_of_the_movie]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:70]
    movie_indices = [i[0] for i in sim_scores]
    a = set(movies_data[['title']].iloc[movie_indices]['title'])
    return a


def get_recommendations_intersection(movie_name1, movie_name2):
    rec1 = get_recommendations(movie_name1)
    rec2 = get_recommendations2(movie_name2)
    r_final= pd.DataFrame({'title':list(rec1.intersection(rec2))})
    final_suggestion=movies_data[movies_data['title'].isin(r_final['title'].tolist())].drop(['index','budget','homepage','id',
 'keywords',
 'original_language',
 'overview',
 'popularity',
 'production_companies',
 'release_date',
 'revenue',
 'runtime','production_countries','original_title',
 'spoken_languages',
 'status',
 'tagline',
 'vote_average',
 'vote_count',
 'crew'],1).reset_index(drop=True)
    if  len(final_suggestion)==0:
        print('sorry, you should select two similar movie')
    else:
        return final_suggestion


# In[45]:


get_recommendations_intersection('inception','interstaller')


# # use streamlit library to creat web app
# 

# In[ ]:


import streamlit as st
import requests
from streamlit_lottie import st_lottie


# In[14]:


st.subheader('Hi Im Ali Nadi Khorasgani' )
st.title('This is a movie recommendation system Web App')


# In[15]:


def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
lottie_coding= load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_khzniaya.json')
st_lottie(lottie_coding,height=300,key='coding')


# In[ ]:





# In[16]:


st.image('abcd.jpeg')

st.markdown("---")





# In[17]:


movies_list = movies_data['title'].values

selected_movie1 = st.selectbox( " select your first favorite movie", movies_list,key=1 )
selected_movie2 = st.selectbox( "select your first favorite movie", movies_list,key=2 )
st.write('we suggest you these movies ')
if st.button('Show Recommendation'):
      recommended_movie_names = get_recommendations_intersection(selected_movie1,selected_movie2)
      st.write(recommended_movie_names)



st.markdown("---")


st.markdown('[download_dataset](https://drive.google.com/file/d/1cCkwiVv4mgfl20ntgY3n4yApcWqqZQe6/view)')

