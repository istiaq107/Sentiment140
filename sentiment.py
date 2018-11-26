import pdb
import csv
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

with open("data.csv", "r") as file:
    data = np.array(list(csv.reader(file)))
    
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(data[:, 5], data[:, 0], test_size=0.1, random_state=42)

le = preprocessing.LabelEncoder()
le.fit(labels_train)
labels_train = le.transform(labels_train)
labels_test = le.transform(labels_test)

vectorizer = TfidfVectorizer()
features_train_transformed = vectorizer.fit_transform(features_train).toarray()
features_test_transformed  = vectorizer.transform(features_test).toarray()

gnb = GaussianNB()
gnb.fit(features_train_transformed, labels_train)
predicted = gnb.predict(features_test_transformed)

print(accuracy_score(predicted, labels_test))
pdb.set_trace()