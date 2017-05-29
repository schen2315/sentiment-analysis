from bs4 import BeautifulSoup
import numpy as np  
import re
import json

def tweet_to_words(tweet):

    # 1. Remove HTML
    tweet_text = BeautifulSoup(tweet, "lxml").get_text() 
    # remove @usr & only use words
    letters_only = re.sub("@[a-zA-Z0-9]+"," ", tweet_text)
    letters_only = re.sub("[^a-zA-Z0-9]", " ", letters_only)
    words = letters_only.lower().split()  
    return(" ".join(words))

# not necessary but maybe I'll use it...
def tweets2wordset(tweets, wordset):
	wordset = {}
	index = 0
	for i in range(0, len(cleaned_tweets)):
		split = cleaned_tweets[i].split(" ")
		for j in range(0, len(split)):
			if split[j] not in wordset:
				wordset[split[j]] = index
				print(split[j], index)
				index = index + 1
	return wordset, index

def sanitize(tweets, nfeatures, wordset):
	my_input = []
	for i in range(0, len(tweets)):
		my_input.append(tweet2features(tweet_to_words(tweets[i]), nfeatures, wordset))
	return my_input
def tweet2features(tweet, nfeatures, wordset):
	# observation = [0]*nfeatures
	observation = np.zeros(nfeatures)
	#print("number of features", len(observation))
	split = tweet.split(' ')
	for i in range(0, len(split)):
		if split[i] in wordset:
			observation[wordset[split[i]]] += 1
	# return np.asarray(observation).reshape(1, -1)
	return observation
def sentiment2class(x):
	if(x == 4):
		return 1
	elif(x == 2):
		return 0
	elif(x == 0):
		return -1
	else:
		raise ValueError('invalid sentiment class, must be either 4,2, or 0')

def convertova(x, cl):
	if(x == cl):
		return 1
	else:
		return 0
def num2class_name(x):
	if(x == 1):
		return "positive sentiment"
	elif(x == 0): 
		return "neutral sentiment"
	else:
		return "negative sentiment"