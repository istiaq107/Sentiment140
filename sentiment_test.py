import re
import pdb
import string
from sklearn.externals import joblib
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer

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

clf = joblib.load("sentiment.sav")
vectorizer = joblib.load("vectorizer.sav")

def predict(input):
    vector = vectorizer.transform([preprocess_tweets(input)])
    return clf.predict(vector)

pdb.set_trace()