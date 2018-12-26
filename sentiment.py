import pdb
import csv
import re
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def preprocess_tweets(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

train_data = pd.read_csv("data.csv")
train_data = train_data[:15000]
print "data acquired."
train_data['tweet'] = train_data['tweet'].apply(preprocess_tweets)
train_data.drop('id', inplace=True, axis=1)
print "data preprocessing complete."

features_train, features_test, labels_train, labels_test = train_test_split(train_data['tweet'], train_data['label'], test_size=0.2, random_state=3)
print "features and labels split into training and testing sets."
Vectorizer = TfidfVectorizer(stop_words='english', encoding="ISO-8859-1", sublinear_tf=True)

features_train_tfidf = Vectorizer.fit_transform(features_train)
features_test_tfidf = Vectorizer.transform(features_test)
print "features vectorized"

clf = RandomForestClassifier()
clf.fit(features_train_tfidf, labels_train)
print "classifier training complete."
predicted = clf.predict(features_test_tfidf)

print "Accuracy Score: ", accuracy_score(predicted, labels_test)
