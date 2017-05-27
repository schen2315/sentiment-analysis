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
import sanitize

train = pd.read_csv("twitter.csv", header=0, 
                    delimiter=",", quoting=1) 	#QUOTE_ALL

# print(train);

nfeatures = 0
wordset = {}


def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
	print("training model ...")
	model.fit( train_x, train_y)
	p = model.predict_proba( test_x )	# what does p[:, 0] represent?
	# print("Probability on test set: \n", p[:,1])	# probability estimate for the class
	auc = AUC( test_y, p[:,1] )	
	return auc

num_tweets = train["tweet"].size;
print("Cleaning %d tweets ..." % (num_tweets));
cleaned_tweets = []
for i in range(0, num_tweets):
	if(i%100 == 0):
		print("Tweet %d of %d\n" % ( i+1, num_tweets))
	cleaned_tweets.append(sanitize.tweet_to_words(train["tweet"][i]))

train_sentiment = np.asarray(list(map(sanitize.sentiment2class, train["sentiment"])))


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
cleaned_sample = sanitize.tweet_to_words(sample_tweet)

sample_observ = sanitize.tweet2features(cleaned_sample, nfeatures, wordset)
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
	trainovay = np.asarray(list(map(sanitize.convertova, y_train, [i]*len(y_train))))
	testovay = np.asarray(list(map(sanitize.convertova, y_test, [i]*len(y_test))))
	auc = train_and_eval_auc(lr, trainovax, trainovay, testovax, testovay)
	print("logistic regression AUC of class %s: " % sanitize.num2class_name(i), auc)


# persist the model
lr.fit(x_train, y_train)
joblib.dump(lr, 'twitter_sentiment.pkl')

feature_names = vectorizer.get_feature_names()
with open("feature_names.json", "w") as outfile:
	json.dump(feature_names, outfile)




