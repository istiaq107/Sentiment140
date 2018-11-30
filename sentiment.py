import pdb
import csv
import re
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def drop_features(features,data):
    data.drop(features,inplace=True,axis=1)

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

train_data = pd.read_csv("train.csv")
print "data acquired."
train_data['SentimentText'] = train_data['SentimentText'].apply(process_tweet)
drop_features(['ItemID'],train_data)
print "data cleaned."

features_train, features_test, labels_train, labels_test = train_test_split(train_data['SentimentText'], train_data['Sentiment'], test_size=0.2, random_state=42)
print "features and labels split into training and testing sets."

count_vect = CountVectorizer(stop_words='english', encoding="ISO-8859-1")
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
features_count = count_vect.fit_transform(features_train)
features_tfidf = transformer.fit_transform(features_count)
pdb.set_trace()


vectorizer = TfidfVectorizer(encoding="ISO-8859-1")
features_train_transformed = vectorizer.fit_transform(features_train).toarray()
features_test_transformed  = vectorizer.transform(features_test).toarray()
print "Feature vectorization completed."


gnb = GaussianNB()
gnb.fit(features_train_transformed, labels_train)
print "Classifier training complete."
predicted = gnb.predict(features_test_transformed)

print "Accuracy Score: ", accuracy_score(predicted, labels_test)
