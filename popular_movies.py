import pandas as pd
import pickle

ratings = pd.read_csv("./MusicRecommendation/ratings.csv")


print(ratings['movieId'].value_counts()[0:800].index.tolist())

pickle.dump(  ratings['movieId'].value_counts()[0:800].index.tolist()  ,open("popular.p",'wb'))
