import pdb
import csv
import re
import string
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


########### Data Acquisition and Pre-processing ###########
def preprocess_tweets(tweet):
    tweet = re.sub(r'\&\w*;', '', tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r'#\w*', '', tweet)
    tweet = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', tweet)
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = [char for char in list(tweet) if char not in string.punctuation]
    tweet = ''.join(tweet)
    tweet = tweet.lstrip(' ') 
    return tweet

train_data = pd.read_csv("big_data.csv")
print "data acquired."
train_data['SentimentText'] = train_data['SentimentText'].apply(preprocess_tweets)
print "data preprocessing complete."

features_train, features_test, labels_train, labels_test = train_test_split(train_data['SentimentText'], train_data['Sentiment'], test_size=0.2, random_state=3)
print "features and labels split into training and testing sets."


########### Data Vectorization ###########
Vectorizer = TfidfVectorizer(stop_words='english', encoding="ISO-8859-1", lowercase=True, sublinear_tf=True)
features_train_tfidf = Vectorizer.fit_transform(features_train)
features_test_tfidf = Vectorizer.transform(features_test)
print "features vectorized"


########### Classifier Fitting/Dumping ###########
clf = ComplementNB()
clf.fit(features_train_tfidf, labels_train)

joblib.dump(clf, "sentiment.sav")
joblib.dump(Vectorizer, "vectorizer.sav")


########### Prediction ###########
predicted = clf.predict(features_test_tfidf)
print "Accuracy Score: ", accuracy_score(predicted, labels_test)
