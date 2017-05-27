from bs4 import BeautifulSoup 
import pandas as pd     
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import json

train = pd.read_csv("twitter.csv", header=0, 
                    delimiter=",", quoting=1) 	#QUOTE_ALL

# print(train);

nfeatures = 0
wordset = {}


def tweet_to_words(tweet):

    # 1. Remove HTML
    tweet_text = BeautifulSoup(tweet, "lxml").get_text() 
    # remove @usr & only use words
    letters_only = re.sub("@[a-zA-Z0-9]+"," ", tweet_text)
    letters_only = re.sub("[^a-zA-Z0-9]", " ", letters_only)
    words = letters_only.lower().split()  
    return(" ".join(words))

# not necessary but maybe I'll use it...
def tweets2wordset(tweets):
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


def tweet2features(tweet):
	observation = [0]*nfeatures
	print("number of features", len(observation))
	split = tweet.split(' ')
	for i in range(0, len(split)):
		if split[i] in wordset:
			observation[wordset[split[i]]] += 1
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


def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
	print("training model ...")
	model.fit( train_x, train_y)
	p = model.predict_proba( test_x )	# what does p[:, 0] represent?
	# print("Probability on test set: \n", p[:,1])	# probability estimate for the class
	auc = AUC( test_y, p[:,1] )	
	return auc

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

num_tweets = train["tweet"].size;
print("Cleaning %d tweets ..." % (num_tweets));
cleaned_tweets = []
for i in range(0, num_tweets):
	if(i%100 == 0):
		print("Tweet %d of %d\n" % ( i+1, num_tweets))
	cleaned_tweets.append(tweet_to_words(train["tweet"][i]))

train_sentiment = np.asarray(list(map(sentiment2class, train["sentiment"])))


# I'll need to map each word manually to its index
# Maybe later try to pipe the data



vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 


train_data_features = vectorizer.fit_transform(cleaned_tweets)

train_data_features = train_data_features.toarray()

wordset = vectorizer.vocabulary_
nfeatures = len(train_data_features[0])
print("number of features", nfeatures)


# sample_tweet = "&#39;@TyDButler How is Kobe being completely ignored in the Lebron vs MJ discussion.&#39;"
sample_tweet = "Nah but if he 27 I can&#39;t believe he isn&#39;t saying Kobe instead. https://t.co/Ki3sqiRuiC&quot;"
cleaned_sample = tweet_to_words(sample_tweet)

sample_observ = tweet2features(cleaned_sample)
print(sample_observ)


'''
# normalizing makes (+) & (-) but (0) worse
train_data_features = preprocessing.normalize(train_data_features, axis=0)


# applying PCA 
pca = PCA()
print(pca.fit(train_data_features))
# print(pca.explained_variance_ratio_) 
train_data_features = pca.fit_transform(train_data_features)
'''

'''
# positive does better, negative and neutral do worse
train_data_features = preprocessing.normalize(train_data_features)
# axis 1 scales each sample
# this increases AUC for (+) and neutral but decreases for the (-) classes
train_data_features = preprocessing.maxabs_scale(train_data_features, axis = 1, copy = False)

ma = preprocessing.MaxAbsScaler(copy = False)
train_data_features = ma.fit_transform(train_data_features, train_sentiment)


# doing scaling makes AUC score significantly worse
train_data_features = preprocessing.scale(train_data_features)
'''


x_train, x_test, y_train, y_test = train_test_split( \
	train_data_features, train_sentiment, test_size=0.2, random_state=0)

lr = LR(multi_class='ovr')

# do AUC for one vs. all



classes = [1, 0, -1];


for i in classes:
	trainovax = x_train
	testovax = x_test
	trainovay = np.asarray(list(map(convertova, y_train, [i]*len(y_train))))
	testovay = np.asarray(list(map(convertova, y_test, [i]*len(y_test))))
	auc = train_and_eval_auc(lr, trainovax, trainovay, testovax, testovay)
	print("logistic regression AUC of class %s: " % num2class_name(i), auc)

lr.fit(x_train, y_train)
joblib.dump(lr, 'twitter_sentiment.pkl')

# print(lr.predict(np.asarray(sample_observ).reshape(1,-1)))


feature_names = vectorizer.get_feature_names()
with open("feature_names.json", "w") as outfile:
	json.dump(feature_names, outfile)




