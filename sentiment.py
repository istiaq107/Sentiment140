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
print "Data acquired."

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(data[:, 5], data[:, 0], test_size=0.1, random_state=42)
print "Features and labels split into training and testing sets."
pdb.set_trace()

features_train, features_test, labels_train, labels_test = features_train[:32000], features_test[:32000], labels_train[:32000], labels_test[:32000]
print "Testing and training sets reduced in size."

le = preprocessing.LabelEncoder()
le.fit(labels_train)
labels_train = le.transform(labels_train)
labels_test = le.transform(labels_test)
pdb.set_trace()

vectorizer = TfidfVectorizer(encoding="ISO-8859-1")
features_train_transformed = vectorizer.fit_transform(features_train).toarray()
features_test_transformed  = vectorizer.transform(features_test).toarray()
pdb.set_trace()
gnb = GaussianNB()
gnb.fit(features_train_transformed, labels_train)
predicted = gnb.predict(features_test_transformed)

print(accuracy_score(predicted, labels_test))
pdb.set_trace()