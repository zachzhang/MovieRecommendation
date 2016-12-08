

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

features = np.loadtxt("movie_rep.txt")

sample = np.random.permutation(features.shape[0])

features = features[sample]

#model = TSNE(n_components=2, random_state=0)

model = PCA(n_components=2, random_state=0,whiten=True)

low_d = model.fit_transform(features)

np.savetxt("pca_embedding.txt",low_d) 
