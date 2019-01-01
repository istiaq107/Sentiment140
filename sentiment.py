import pdb
import csv
import re
import nltk
import numpy as np
import pandas as pd
from timeit import timeit
from string import punctuation
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


########### Data Pre-processing ###########
def preprocess_tweets(tweet):
    tweet = re.sub(r'\&\w*;', '', tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r'#\w*', '', tweet)
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = [char for char in list(tweet) if char not in punctuation]
    tweet = ''.join(tweet)
    tweet = tweet.lstrip(' ') 
    return tweet

def tokenize_tweets(tweet):
    tokens = word_tokenize(tweet)
    return [w for w in tokens if not w in stopwords.words('english')]

train_data = pd.read_csv("all/train.csv")
# train_data = train_data[:5000]
print "data acquired."
train_data['SentimentText'] = train_data['SentimentText'].apply(preprocess_tweets)
# train_data['tokens'] = train_data['SentimentText'].apply(tokenize_tweets)
train_data.drop('ItemID', inplace=True, axis=1)
print "data preprocessing complete."
# pdb.set_trace()

features_train, features_test, labels_train, labels_test = train_test_split(train_data['SentimentText'], train_data['Sentiment'], test_size=0.2, random_state=3)
print "features and labels split into training and testing sets."

########### Data Vectorization ###########

Vectorizer = TfidfVectorizer(stop_words='english', encoding="ISO-8859-1", lowercase=True, sublinear_tf=True)
features_train_tfidf = Vectorizer.fit_transform(features_train)
features_test_tfidf = Vectorizer.transform(features_test)
print "features vectorized"
# pdb.set_trace()

########### Classifier Fitting ###########
start = timeit()
clf = ComplementNB()
clf.fit(features_train_tfidf, labels_train)

########### Prediction ###########
def clf_predict(string):
    string = preprocess_tweets(string)
    string_tfidf = Vectorizer.transform([string])
    return clf.predict(string_tfidf)

# pdb.set_trace()
print("classifier training complete in %s seconds." % (start - timeit()))
predicted = clf.predict(features_test_tfidf)

print "Accuracy Score: ", accuracy_score(predicted, labels_test)
