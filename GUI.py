from tkinter import ttk

import pandas as pd
import numpy as np
from io import StringIO
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from tkinter import *
from tkinter.ttk import *
import pickle
import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

from surprise.model_selection import KFold
import cloudpickle
import warnings;

# CBF MODEL
"""
df = pd.read_csv("new_movie.csv")
df = df[['title','genre','director','actors','description','imdb_title_id']]
df['Key_words'] = ""
df['bag_of_words'] = ""

def preprocessing():
    for index, row in df.iterrows():
        description = row['description']
        r = Rake()
        r.extract_keywords_from_text(description)
        key_words_dict_scores = r.get_word_degrees()
        row['Key_words'] = list(key_words_dict_scores.keys())
    words = ""
    for item in row['Key_words']:
        words = words + (str)(item) + " "
    # for item in row['actors']:
    # words = words + (str)(item) + " "
    # for item in row['director']:
    # words = words + (str)(item) + " "
    words += (str)(row['title'])
    words += (str)(row['genre'])
    row['bag_of_words'] = words
preprocessing()
def build():
    df.drop(columns=['description'], inplace=True)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    matrix = tf.fit_transform(df['bag_of_words'])
    matrix.astype(np.float32)
    global cosine_sim,indices
    cosine_sim= cosine_similarity(matrix.astype(np.float32))
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    with open('cbf.pkl', 'wb') as output:
        pickle.dump(cosine_sim, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(indices, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)


build()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


"""

# print(get_recommendations('Wonder Woman'))

with open('cbf.pkl', 'rb') as input:
    cosine_sim = pickle.load(input)
    indices = pickle.load(input)
    df = pickle.load(input)


    def get_recommendations_cbf(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        with open('id.pkl', 'wb') as output:
            res = df['imdb_title_id'].iloc[movie_indices]
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(title, output, pickle.HIGHEST_PROTOCOL)
        return df['title'].iloc[movie_indices]


# CF MODEL

def get_recommendation_cf(movie_name):
    movies = pd.read_csv("movies_cleaned.csv")
    ratings = pd.read_csv("ratings1.csv")
    ratings = ratings.head(70000)
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
    final_dataset.fillna(0, inplace=True)
    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
    final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
    final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]
    sample = np.array([[0, 0, 3, 0, 0], [4, 0, 0, 0, 2], [0, 0, 0, 0, 1]])
    csr_sample = csr_matrix(sample)
    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
        return df
    else:
        return "No movies found. Please check your input"


# print(get_recommendation_cf('Iron Man'))


# HYBRID MODEL

def hybrid():
    m_df = pd.read_csv('movies_metadata.csv')
    m_df['genres'] = m_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x]
    if isinstance(x, list) else [])
    # Claculation of c
    vote_counts = m_df[m_df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = m_df[m_df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)
    m_df['year'] = pd.to_datetime(m_df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0]
    if x != np.nan else np.nan)
    col_list = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']
    qualified = m_df[(m_df['vote_count'] >= m)
                     & (m_df['vote_count'].notnull())
                     & (m_df['vote_average'].notnull())][col_list]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)
    temp = m_df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    temp.name = 'genre'
    mgen_df = m_df.drop('genres', axis=1).join(temp)

    ## Content Based Recommendation model

    small_mdf = pd.read_csv('links_small.csv')
    small_mdf = small_mdf[small_mdf['tmdbId'].notnull()]['tmdbId'].astype('int')

    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan

    m_df['id'] = m_df['id'].apply(convert_int)
    m_df = m_df.drop([19730, 29503, 35587])
    m_df['id'] = m_df['id'].astype('int')
    sm_df = m_df[m_df['id'].isin(small_mdf)]
    sm_df['tagline'] = sm_df['tagline'].fillna('')
    sm_df['description'] = sm_df['overview'] + sm_df['tagline']
    sm_df['description'] = sm_df['description'].fillna('')
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(sm_df['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    sm_df = sm_df.reset_index()
    titles = sm_df['title']
    indices = pd.Series(sm_df.index, index=sm_df['title'])

    credits = pd.read_csv('credits.csv')
    keywords = pd.read_csv('keywords.csv')

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    m_df['id'] = m_df['id'].astype('int')

    m_df = m_df.merge(credits, on='id')
    m_df = m_df.merge(keywords, on='id')

    sm_df = m_df[m_df['id'].isin(small_mdf)]

    sm_df['cast'] = sm_df['cast'].apply(literal_eval)
    sm_df['crew'] = sm_df['crew'].apply(literal_eval)
    sm_df['keywords'] = sm_df['keywords'].apply(literal_eval)
    sm_df['cast_size'] = sm_df['cast'].apply(lambda x: len(x))
    sm_df['crew_size'] = sm_df['crew'].apply(lambda x: len(x))

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    sm_df['director'] = sm_df['crew'].apply(get_director)
    sm_df['cast'] = sm_df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    sm_df['cast'] = sm_df['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    sm_df['keywords'] = sm_df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    sm_df['cast'] = sm_df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    sm_df['director'] = sm_df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    sm_df['director'] = sm_df['director'].apply(lambda x: [x, x])
    s = sm_df.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    s = s.value_counts()
    s = s[s > 1]
    stemmer = SnowballStemmer('english')
    stemmer.stem('dogs')

    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words

    sm_df['keywords'] = sm_df['keywords'].apply(filter_keywords)
    sm_df['keywords'] = sm_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    sm_df['keywords'] = sm_df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    sm_df['soup'] = sm_df['keywords'] + sm_df['cast'] + sm_df['director'] + sm_df['genres']
    sm_df['soup'] = sm_df['soup'].apply(lambda x: ' '.join(x))
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(sm_df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    sm_df = sm_df.reset_index()
    titles = sm_df['title']
    indices = pd.Series(sm_df.index, index=sm_df['title'])

    def get_recommendations_cbf(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:51]
        movie_indices = [i[0] for i in sim_scores]

        movies = sm_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[
            (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(10)
        return qualified

    # print(get_recommendations_cbf('The Dark Knight'))

    # print(get_recommendations_cbf('Pulp Fiction'))

    ## Collaborative Filtering

    reader = Reader()
    ratings = pd.read_csv('ratings_small.csv')
    ratings.head()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    kf = KFold(n_splits=5)
    kf.split(data)
    svd = SVD()
    # evaluate(svd, data, measures=['RMSE', 'MAE'])
    # cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # svd.predict(1, 302, 3)

    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan

    id_map = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(sm_df[['title', 'id']], on='id').set_index('title')
    indices_map = id_map.set_index('id')

    def get_recommendations_hybrid(userId, title):
        idx = indices[title]
        tmdbId = id_map.loc[title]['id']
        # print(idx)
        movie_id = id_map.loc[title]['movieId']
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = sm_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)

    print(get_recommendations_hybrid(500, 'Avatar'))


if __name__ == "__main__":
    # print(get_recommendations_cbf('Wonder Woman'))
    root = Tk()
    root.title('GUI for Hybrid movie RS - Final year project')
    root.geometry("450x700")
    # btn1 = Button(root, text='Quit!', command=root.destroy).place(x=520, y=2)
    style = ttk.Style()
    style.configure("BW.TLabel", background="red",foreground="white")
    style.configure("BW.RLabel", background="red",foreground="white")
    x=Label(root, text="Select Recommendation engine :",style="BW.TLabel").place(x=10, y=10)
    var = IntVar()
    def sel():
        selection = StringVar()
        if var.get() == 1:
            selection = 'RECOMMENDATION ENGINE SELECTED : Content based filtering'
            Button(root, text="BUILD MODEL", command=build1).place(x=15, y=110)
            rs = 1
        elif var.get() == 2:
            selection = "RECOMMENDATION ENGINE SELECTED : Collaborative filtering"
            Button(root, text="BUILD MODEL", command=build2).place(x=15, y=110)
            rs = 2
        else:
            selection = "RECOMMENDATION ENGINE SELECTED : HYBRID recommendation system"
            Button(root, text="BUILD MODEL", command=build3).place(x=15, y=110)
            rs = 3
        rs = Message(root, textvariable=var, width=500).place(x=15, y=70)
        var.set(selection)


    R1 = Radiobutton(root, text="CBF", variable=var, value=1, command=sel).place(x=230, y=10)
    R2 = Radiobutton(root, text="CF", variable=var, value=2, command=sel).place(x=280, y=10)
    R3 = Radiobutton(root, text="Hybrid", variable=var, value=3, command=sel).place(x=330, y=10)


    def build1():
        time.sleep(1)
        Label(root, text="build started....").place(x=15, y=140)
        root.update()
        time.sleep(2)
        Label(root, text="build complete....").place(x=15, y=160)
        root.update()
        time.sleep(1)
        cbf()


    def build2():
        time.sleep(1)
        Label(root, text="build started....").place(x=15, y=140)
        root.update()
        time.sleep(2)
        Label(root, text="build complete....").place(x=15, y=160)
        root.update()
        time.sleep(1)
        cf()


    def build3():
        time.sleep(1)
        Label(root, text="build started....").place(x=15, y=140)
        root.update()
        time.sleep(2)
        Label(root, text="build complete....").place(x=15, y=160)
        root.update()
        time.sleep(1)
        hybrid()


    def cbf():
        name_var = StringVar()

        def submit():
            movie_name = name_var.get()
            print("The movie name is : " + movie_name)
            try:
                with open('cbf1.pkl', mode='rb') as file:
                    instance = cloudpickle.load(file)
                    #res = instance('Avatar')
                    res = instance(movie_name)
                    print(res)
                    res = instance(movie_name)
                    msg = Message(root, text=res).place(x=10, y=350)
                Label(root, text="Recommendations for " + movie_name + " are : ", font=('calibre', 10, 'bold')).place(
                    x=10, y=300)
            except:
                Label(root, text="Wrong movie name", font=('calibre', 10, 'bold')).place(x=10, y=290)
                name_var.set("")

        name_label = Label(root, text='Enter movie name :', font=('calibre', 10, 'bold')).place(x=10, y=200)
        name_entry = Entry(root, textvariable=name_var, font=('calibre', 10, 'normal')).place(x=10, y=230)
        sub_btn = Button(root, text='Submit', command=submit).place(x=10, y=270)


    def hybrid():
        name_var = StringVar()
        id_var = StringVar()

        def submit():
            movie_name = name_var.get()
            movie_id = id_var.get()
            print("The movie name iss : " + movie_name)
            print("The movie id iss : " + movie_id)
            try:
                with open('hybrid.pkl', mode='rb') as file:
                    instance = cloudpickle.load(file)
                    #res = instance(500, 'Avatar')
                    res = instance(movie_id,movie_name)
                    print(res)
                    msg = Message(root, text=res).place(x=10, y=420)
                Label(root, text="Recommendations for " + movie_name + " are : ", font=('calibre', 10, 'bold')).place(
                    x=10, y=380)
            except:
                Label(root, text="Wrong movie name", font=('calibre', 10, 'bold')).place(x=10, y=380)
                name_var.set("")

        name_label = Label(root, text='Enter movie name :', font=('calibre', 10, 'bold')).place(x=10, y=200)
        name_entry = Entry(root, textvariable=name_var, font=('calibre', 10, 'normal')).place(x=10, y=230)
        name_label = Label(root, text='Enter user id :', font=('calibre', 10, 'bold')).place(x=10, y=260)
        name_entry = Entry(root, textvariable=id_var, font=('calibre', 10, 'normal')).place(x=10, y=290)
        sub_btn = Button(root, text='Submit', command=submit).place(x=10, y=330)


    def cf():
        name_var = StringVar()

        def submit():
            movie_name = name_var.get()
            print("The movie name is : " + movie_name)
            try:
                res = get_recommendation_cf(movie_name)
                print(res)
                msg = Message(root, text=res).place(x=10, y=350)
                Label(root, text="Recommendations for " + movie_name + " are : ", font=('calibre', 10, 'bold')).place(
                    x=10, y=300)
            except:
                Label(root, text="Wrong movie name", font=('calibre', 10, 'bold')).place(x=10, y=290)
                name_var.set("")

        name_label = Label(root, text='Enter movie name :', font=('calibre', 10, 'bold')).place(x=10, y=200)
        name_entry = Entry(root, textvariable=name_var, font=('calibre', 10, 'normal')).place(x=10, y=230)
        sub_btn = Button(root, text='Submit', command=submit).place(x=10, y=270)


    root.mainloop()
