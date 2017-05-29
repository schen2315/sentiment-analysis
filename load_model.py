from sklearn.externals import joblib
import json
import sanitize

sample_tweet = ["Nah but if he 27 I can&#39;t believe he isn&#39;t saying Kobe instead. https://t.co/Ki3sqiRuiC&quot;"]

features = []
wordset = {}

def predict(tweets, model_file, features_file):
	model = joblib.load(model_file)

	with open(features_file, "r") as infile:
		features = json.load(infile)

	nfeatures = len(features)
	for i in range(0, nfeatures):
		wordset[features[i]] = i

	sample_observ = sanitize.sanitize(tweets, nfeatures, wordset)
	prediction = model.predict(sample_observ)
	return prediction

prediction = predict(sample_tweet, "twitter_sentiment.pkl", "feature_names.json")
print(prediction)