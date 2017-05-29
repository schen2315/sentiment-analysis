from sklearn.externals import joblib
import json
import sanitize

model = joblib.load('twitter_sentiment.pkl')
sample_tweet = "Nah but if he 27 I can&#39;t believe he isn&#39;t saying Kobe instead. https://t.co/Ki3sqiRuiC&quot;"

features = []

with open("feature_names.json", "r") as infile:
	features = json.load(infile)

nfeatures = len(features)
wordset = {}
for i in range(0, nfeatures):
	wordset[features[i]] = i

def predict(tweet):
	cleaned_tweet = sanitize.tweet_to_words(tweet)
	sample_observ = sanitize.tweet2features(cleaned_tweet, nfeatures, wordset)
	return model.predict(sample_observ)

