# Sentiment140

## Overview
Using sklearn's ComplementNB classifier, I developed a sentiment classifier that trains using [Sentiment140](https://www.kaggle.com/kazanova/sentiment140), 
a Twitter sentiment dataset of 1.6 million records. The current rendition of the classifier only classifies between positive 
and negative text sentiments. It's not very reliable when passing neutral texts, which I do plan to solve later when a dataset
including neutral sentiment shows up in my research of Sentiment Analysis. 


## Data Pre-Processing for Classifier
Because the used dataset is of twitter tweets, they tend to contain lots of junk, such as hashtags, emojis, user tags, html links, punctuation etc, which are all
stripped off before the data is used to train as they don't add any value while deducing the sentiment of a text. Next, a TF-IDF vectorizer is fitted with the training 
dataset to record the term-frequencies of the words in text, which then transforms the dataset of words into a sparse matrix of TF-IDF corpus.
This sparse matrix is then used to train the NB classifier.


## Accuracy Disclaimer
This classifier has an accuracy rate of around 76% which I've spent countless hours trying to improve. After trying other classifiers, 
and even ensemble learning, I found out that ComplementNB has the highest accuracy rate, and in one article I noticed something interesting 
which led me to the conclusion that with my current knowledge of sentiment analysis, it's not possible to get higher. Even when humans are 
reading texts, there would be a 20% uncertainty as to what sentiment said text is expressing, so with the available records we have on 
sentiment, it's not possible for a machine-learning model to have over a 95% accuracy(at least not with the knowledge I have.)


## Trying it out
I plan to deploy it in my website for people to play around with it, and test it out, but that's still under construction, so for now
to test it out, I've dumped the vectorizer and the classifier models, and written a program that loads those models. To call the program
run the following commands(everything's in Python2.7):

1.`git clone https://github.com/istiaq107/Sentiment140`

2.`pip install -r requirements.txt`

3.`python sentiment_test.py`

The debugger is activated, and to test, pass the text as an input parameter to the function `predict`. If it returns 0, it means negative,
if 1, positive.
