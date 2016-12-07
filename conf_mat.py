import numpy as np
from sklearn.metrics import confusion_matrix

yhat = np.loadtxt("./MusicRecommendation/y_hat.txt") +3.52
y = np.loadtxt("./MusicRecommendation/y.txt") +3.52

print(y[0:10])

yhat_bin = np.digitize(yhat,np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))

y_bin = np.digitize(y,np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))


cm = confusion_matrix(y_bin[:200000].astype(int), yhat_bin.astype(int))

print(cm)

np.savetxt("cm.txt",cm)

