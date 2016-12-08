import numpy as np

import pickle
import heapq

import pandas as pd

x  = np.loadtxt("user_mat.txt")


#our ids and movie lens ids
mov_lens_ids = pd.read_csv("movie_titles.csv")
mov_lens_ids.columns= ["id","movie_lens_id"]

#movie lens ids and titles
titles = pd.read_csv("./MusicRecommendation/movies.csv")

title_map = pd.merge(mov_lens_ids,titles,how='left',left_on ="movie_lens_id",right_on="movieId")


id2title = dict(zip( list(title_map["id"]) , list(title_map["title"])   ) )
ml_id2title = dict(zip( list(title_map["movie_lens_id"]) , list(title_map["title"])   ) )
ml_id2id = dict(zip( list(title_map["movie_lens_id"]) , list(title_map["id"])   ) ) 

ratings = pd.read_csv("./MusicRecommendation/ratings.csv")

ratings_mean = ratings["rating"].mean()


uni = pickle.load(open("uni.p","rb"))
popular_ml_id = pickle.load(open("popular.p","rb"))
popular_id = []
for i in popular_ml_id:
    if i in ml_id2id:
        popular_id.append(ml_id2id[i])

unpopular_ids = set(range(x.shape[1])) - set(popular_id)
unpopular_ids = np.array(list(unpopular_ids)).astype(int)

print(len(popular_id) , x.shape[1] , len(unpopular_ids))

x[:,unpopular_ids] = 0

user_movies = {}

#uni = [uni[11]]


path = "/home/ubuntu/UserRec/"

for j,user in enumerate(uni):

    df = ratings.loc[ratings['userId'] == user]

    movies = []
    movies_r = []
    for i,r in  zip(list(df['movieId']) , list(df['rating'])) :
        if i in ml_id2title:
            movies.append(ml_id2title[i])
            movies_r.append(r)


    user_movies[user] = movies

    #rec_ids = np.argsort(x[j])[-30:]
    

    rec_ids = np.random.permutation( len(popular_id))[0:30]
    rec_ids = list(np.array(popular_id)[rec_ids])

    rec_movies = [ id2title[k] for k in list(rec_ids) ]
    rec_r = [ratings_mean + x[j][k] for k in list(rec_ids) ]  

    user_df = pd.DataFrame([ movies,movies_r ]   ).transpose()
    user_df.columns = ["user_movie","user_movie_rating"]

    rec_df = pd.DataFrame([ rec_movies,rec_r ]   ).transpose() 
    rec_df.columns = ["rec_movie","rec_movie_rating"]

    user_df.to_csv(path+str(user)+"_movies.csv")
    rec_df.to_csv(path+str(user)+"_recommended.csv") 

   

    print("USER MOVIES:")
    print(list(zip(movies ,movies_r )) )
    print()
    print("RECOMMENDED MOVIES:")
    print(list(zip(rec_movies ,rec_r) ))
    print()


    
