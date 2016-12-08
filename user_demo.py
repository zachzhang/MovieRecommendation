import pandas as pd
import numpy as np
import time
import pickle

#user_movie = pd.read_csv("./MusicRecommendation/user_demo.csv")


#user_movie = np.array(user_movie)

#np.savetxt("user_demo.txt",user_movie )

rating = np.loadtxt("./MusicRecommendation/user_yhat.txt")
user_movie = np.loadtxt("user_demo.txt").astype(int)
user_movie = np.transpose(user_movie)


X = np.zeros((41,np.max(user_movie[:,1]) +1))


uni = list(np.unique(user_movie[:,0]))

pickle.dump(uni,open("uni.p","wb"))


user_map = dict(zip(uni  ,range(len(uni) )))

print(uni)

for i in range(rating.shape[0]):

    X[user_map[user_movie[i,0]] ,user_movie[i,1]] = rating[i]

np.savetxt("user_mat.txt",X)

