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
mask = imread("butterflybig.png")
onlyfiles = np.sort(onlyfiles)
for i, filename in enumerate(onlyfiles[2::2]):
	fig = plt.figure()
	for idx2 in [0, 1]:
		idx = i + idx2
		fil = onlyfiles[idx]
		if 'movies' in fil:
			col = 'user_movie'
		else:
			col = 'rec_movie'
		df = pd.read_csv(mypath + str('/') + fil)
		#print(df)
		#sys.exit(0)



		#Convert all the required text into a single string here 
		#and store them in word_string

		#you can specify fonts, stopwords, background color and other options
		for idx, s in enumerate(df[col]):
			df[col][idx] = df[col][idx].replace (" ", "_")
			df[col][idx] = df[col][idx].replace (",", "_")
			df[col][idx] = df[col][idx].replace ("'", "_")
			df[col][idx] = df[col][idx].replace (".", "_")
			df[col][idx] = df[col][idx].replace ("__", "_")
			df[col][idx] = df[col][idx][:-6]
			while df[col][idx][0] == '_':
				df[col][idx] = df[col][idx][1:]
			while df[col][idx][-1] == '_':
				df[col][idx] = df[col][idx][:-1]

		st = ''
		for s, val in zip(df[col],df[col + '_rating']):
			st += (s + ' ') * int(val)
		word_string = df[col].to_string()
		print(df)
		print(word_string)
		wordcloud = WordCloud(
				          stopwords=STOPWORDS,
				          background_color='black',
				          width=1200,
				          height=1000,
					  mask=mask,
				         ).generate(word_string)
		if 'movies' in fil:
			plt.title('True Ratings')
		else:
			plt.title('Predicted Ratings')
		ax = fig.add_subplot(121 + idx2)
		plt.imshow(wordcloud)
		plt.axis('off')
	#plt.show()
	plt.savefig('wordcloudcomparisons3/' + str(fil) + '.png')
	plt.close()
