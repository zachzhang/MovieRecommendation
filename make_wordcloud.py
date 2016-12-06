import sys
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.misc import imread

mypath='UserRec'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
mask = imread("butterfly.png")

for fil in onlyfiles[1:]:
	if 'movies' in fil or '0' in fil:
		continue
	df = pd.read_csv(mypath + str('/') + fil)
	#print(df)
	#sys.exit(0)



	#Convert all the required text into a single string here 
	#and store them in word_string

	#you can specify fonts, stopwords, background color and other options

	for idx, s in enumerate(df['rec_movie']):
		df['rec_movie'][idx] = df['rec_movie'][idx].replace (" ", "_")
		df['rec_movie'][idx] = df['rec_movie'][idx].replace (",", "_")
		df['rec_movie'][idx] = df['rec_movie'][idx].replace ("'", "_")
		df['rec_movie'][idx] = df['rec_movie'][idx].replace (".", "_")
		df['rec_movie'][idx] = df['rec_movie'][idx].replace ("__", "_")
		df['rec_movie'][idx] = df['rec_movie'][idx][:-6]
		while df['rec_movie'][idx][0] == '_':
			df['rec_movie'][idx] = df['rec_movie'][idx][1:]
		while df['rec_movie'][idx][-1] == '_':
			df['rec_movie'][idx] = df['rec_movie'][idx][:-1]

	st = ''
	for s, val in zip(df['rec_movie'],df['rec_movie_rating']):
		st += (s + ' ') * int(np.exp(val))
	word_string = df['rec_movie'].to_string()
	print(df)
	print(word_string)
	wordcloud = WordCloud(
		                  stopwords=STOPWORDS,
		                  background_color='white',
		                  width=1200,
		                  height=1000,
				  mask=mask,
		                 ).generate(word_string)


	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()
	plt.savefig('wordclouds/' + str(fil) + '.png')
	plt.close()
