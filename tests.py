from sklearn.externals import joblib
import numpy as np
import pandas as pd
import sanitize
import json


test_input = pd.read_csv("test_input.csv", delimiter="\n", dtype="str",header=None)[0]
test_input_sanitized = np.genfromtxt("test_input_sanitized.csv", delimiter=",").astype(int)
model = joblib.load('twitter_sentiment.pkl')


with open("feature_names.json", "r") as infile:
	features = json.load(infile)
nfeatures = len(features)
wordset = {}
def setup():
	for i in range(0, nfeatures):
		wordset[features[i]] = i
def runTests():
	setup()
	testSanitize()
def testSanitize():
	my_input = sanitize.sanitize(test_input, nfeatures, wordset)
	my_output = model.predict(my_input)
	test_output_sanitized = model.predict(test_input_sanitized)
	for i in range(0, len(my_output)):
		assert my_output[i] == test_output_sanitized[i]

runTests()