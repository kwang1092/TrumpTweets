import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from textblob import TextBlob

from collections import defaultdict

import torch
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models import InferSent

from dateutil import parser
from datetime import datetime
import time


'''
Parse time formatted as string like (7/12/2016 0:56) to time categories
'''
def parse_time(cur):
    t1 = parser.parse(cur)
    time = t1.hour * 60 + t1.minute

    #morning 6 am to noon
    if time >=360 and time < 720:
        time_class = 0
    #afternoon noon to 6 pm
    elif time >= 720 and time < 1080:
        time_class = 1
    #evening 6 pm to midnight
    elif time >= 1080 and time < 1440:
        time_class = 2
    # dumb times midnight to 6 am
    else:
        time_class = 3
    return time_class

'''
Removing URLs from tweet text body and returned the new string with boolean indicating
whether there was a link or not
'''
def remove_urls(str):
    ind = str.find("https")
    if ind != -1:
        return str[:ind], 1
    return str, 0

'''
Removing hashtags from tweet text body
'''
def extract_hash_tags(s):
    hashtags = set(part for part in s.split() if part.startswith('#'))
    for hashtag in hashtags:
        s = s.replace(hashtag + " ", '')
        s = s.replace(hashtag, '')
    return s

'''
Removing @mentions from tweet text body
'''
def extract_at(s):
    ats = set(part for part in s.split() if part.startswith('@'))
    for at in ats:
        s = s.replace(at + " ", '')
        s = s.replace(at, '')
    return s

'''
Formed our features
'''
def preprocessing(in_file, is_test = False):
    train_data = pd.read_csv(in_file, encoding='utf-8')
    data = []
    new_labels = []
    ids = []
    opinions = []
    subjectives = []

    texts = train_data['text']
    id = train_data['id']
    if not is_test:
        labels = train_data['label']

    sentence_dict = {}
    sentences = []
    links = []
    hashes = []
    ats = []
    tweet_length = []
    index = 0
    for i in range(len(texts)):
        str = texts[i]
        if '#' in str:
            hashes.append(1)
        else:
            hashes.append(0)
        if '@' in str:
            ats.append(1)
        else:
            ats.append(0)

        str, link = remove_urls(str)
        str = extract_hash_tags(str)
        str = extract_at(str)

        opinion = TextBlob(str)
        opinions.append(opinion.sentiment.polarity)
        subjectives.append(opinion.sentiment.subjectivity)
        links.append(link)

        sentence = sent_tokenize(str)
        total_length = 0
        total_characters = 0
        for sent in sentence:
            sent = word_tokenize(sent)
            for word in sent:
                total_characters += len(word)
            total_length += len(sent)
        if total_length == 0:
            total_length = 1
        tweet_length.append(total_characters/total_length)

        ids.append(id[i])
        if not is_test:
            new_labels.append(labels[i])

    sentence_count = defaultdict(int)
    fav_count = train_data['favoriteCount']
    rt_count = train_data['retweetCount']
    created = train_data['created']

    sentence_vectors = np.zeros(shape=(len(texts),7))
    for i in range(len(texts)):
        time = created[i]
        time = parse_time(time)
        sentence_vectors[i] = [opinions[i], subjectives[i],  links[i], time, hashes[i], ats[i], tweet_length[i]]

    return sentence_vectors, new_labels, ids


def bag_o_words(input, labels, test_data, test_ids):
    X = input
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.3, random_state=420)
    X_test = test_data

    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000]
    max_features = ['auto', None, 'log2']
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, None]
    min_samples_splits = np.linspace(start=0.1, stop=1.0, num=10, endpoint=True)
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_splits,
                   'min_samples_leaf': min_samples_leafs,
                   'bootstrap': bootstrap}

    forest_classifier = RandomForestClassifier()
    classifier = RandomizedSearchCV(estimator = forest_classifier, param_distributions = random_grid, n_iter = 500   , cv = 5, verbose=2, random_state=55, n_jobs = -1)
    classifier.fit(X_train, y_train)
    print(classifier.best_params_)

    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Random Forest random accuracy is: " + str(accuracy))

    # classifier = RandomForestClassifier(n_estimators=100, max_depth=2, max_features='auto', random_state=55)
    classifier = RandomForestClassifier(n_estimators=1000, min_samples_split=0.6, min_samples_leaf=0.2, bootstrap=False, max_depth=14, max_features='auto', random_state=55)

    classifier.fit(X, labels)
    y_pred = classifier.predict(X_test)

    with open('output.csv', 'w') as f:
        f.write('ID,Label\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(test_ids[i], y_pred[i]))

    classifier = LogisticRegression(solver='liblinear')
    classifier.fit(X, labels)
    y_pred = classifier.predict(X_test)

    with open('output2.csv', 'w') as f:
        f.write('ID,Label\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(test_ids[i], y_pred[i]))


if __name__ == '__main__':
    train_data, train_labels, _ = preprocessing('data/train.csv')
    test_data, _, test_ids = preprocessing('data/test.csv', is_test = True)
    bag_o_words(train_data, train_labels, test_data, test_ids)
