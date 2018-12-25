import pdb
import csv
import re
import numpy as np
import pandas as pd

from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def preprocess_tweets(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

train_data = pd.read_csv("train.csv")
train_data = train_data[:7000]
print "data acquired."
train_data['SentimentText'] = train_data['SentimentText'].apply(preprocess_tweets)
train_data.drop('ItemID', inplace=True, axis=1)
print "data preprocessing complete."

features_train, features_test, labels_train, labels_test = train_test_split(train_data['SentimentText'], train_data['Sentiment'], test_size=0.2, random_state=42)
print "features and labels split into training and testing sets."
# pdb.set_trace()
Vectorizer = TfidfVectorizer(stop_words='english', encoding="ISO-8859-1",sublinear_tf=True)

features_train_tfidf = Vectorizer.fit_transform(features_train)
features_test_tfidf = Vectorizer.transform(features_test)
print "features vectorized"
# pdb.set_trace()

# plt.scatter(features_train_tfidf.toarray(), labels_train, color='red')

gnb = ComplementNB(alpha=1.0)
gnb.fit(features_train_tfidf, labels_train)
print "classifier training complete."
predicted = gnb.predict(features_test_tfidf.toarray())

# plt.plot(features_train_tfidf, predicted, color='blue')
# plt.show()

print "Accuracy Score: ", accuracy_score(predicted, labels_test)
