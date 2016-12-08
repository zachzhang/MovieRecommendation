import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


xpad = 0
movieratings = pd.read_csv('ratings.csv')
fig = plt.figure(figsize = (18, 18))
plt.subplot(311)
movieratings['rating'].hist(grid = False)
plt.title('Distribution of Rating Values', y = .89)
plt.xlabel('Rating Value', labelpad=xpad)
plt.ylabel('Number of Occurances')
ax = plt.gca()
ax.set_xlim([0.5, 5])
plt.subplot(312)
ratings = movieratings['movieId'].value_counts()
vect = np.array([(x**3)*100 for x in range(9)])
plt.hist(ratings, bins = vect)
ax = plt.gca()
ax.grid(False)
ax.set_ylim([0, 3000])
plt.title('Ratings Per Movie', y = .89)
plt.xlabel('Number of Ratings per Movie (bins growing with the cube of ratings for visualization)', labelpad=xpad)
plt.ylabel('Occurances (clipped at the top)')
plt.subplot(313)
ratings = movieratings['userId'].value_counts()
print(np.min(ratings))
vect = np.array([(x**2)*10 for x in range(10)])
plt.hist(ratings, bins = vect)
ax = plt.gca()
ax.grid(False)
#ax.set_ylim([0, 3000])
plt.title('Ratings Per user', y = .89)
plt.xlabel('Number of Ratings per User (bins growing with the square of ratings for visualization)', labelpad=xpad)
plt.ylabel('Occurances')
fig.set_dpi(100)
#plt.savefig('histograms.png')
plt.show()
print("i did it")
