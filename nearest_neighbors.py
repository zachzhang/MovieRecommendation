import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import heapq
import bottleneck

mov_lens_ids = pd.read_csv("movie_titles.csv")
mov_lens_ids.columns= ["id","movie_lens_id"]
#features = np.loadtxt("movie_rep.txt")

titles = pd.read_csv("./MusicRecommendation/movies.csv")

neighbors = np.loadtxt("nearest_neigh.txt")

print(titles.keys())
print(titles.shape)

title_map = pd.merge(mov_lens_ids,titles,how='left',left_on ="movie_lens_id",right_on="movieId")


print(mov_lens_ids.shape)

print(title_map.shape)

title_map = dict(zip( list(title_map["id"]) , list(title_map["title"])   ) )




#dist = squareform(pdist(features, 'euclidean'))

#n_closest = np.zeros((dist.shape[0] , 10))

for i in range(5000):
    
    print( [ title_map[int(x)] for x in neighbors[i,0:5] ] )
    #n_closest[i] = heapq.nsmallest(10, dist[i])
    #n_closest[i] = np.argsort(dist[i])[:10]
    #print(i)

#np.savetxt("nearest_neigh.txt",n_closest)
