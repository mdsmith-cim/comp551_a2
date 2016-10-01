import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# If you dont' have this, follow this guide
# https://spacy.io/docs/#getting-started
#import spacy

class preprocess:

    def __init__(self, train_in = 'data/train_in.csv', train_out = 'data/train_out.csv', test_in = 'data/test_in.csv' ):

        self.categories = ['math', 'cs', 'stat', 'physics']
        self.uniques = None
        self.train_in = train_in
        self.train_out = train_out
        self.test_in = test_in

    def get_data_raw(self, ):

        train = pd. read_csv(self.train_in)
        train_y = pd.read_csv(self.train_out)
        test = pd.read_csv(self.test_in)

        X_train = train['abstract'].values
        X_test = test['abstract'].values
        y_train = train_y['category'].values

        return X_train, y_train, X_test

    def get_data_no_numbers(self):

        X_train, y_train, X_test = self.get_data_raw()

        # Remove anything not a letter
        for i in range(0, X_train.shape[0]):
            X_train[i] = re.sub("[^a-zA-Z]", " ", X_train[i])

        for i in range(0, X_test.shape[0]):
            X_test[i] = re.sub("[^a-zA-Z]", " ", X_test[i])

        return X_train, y_train, X_test


    def convert_string_category_to_num(self, y):

        ser = pd.Series(y)

        labels, uniques = pd.factorize(ser)

        self.uniques = uniques

        return labels

    def convert_num_category_to_string(self, y):

        if self.uniques is None:
            raise Exception('Please convert strings to categories first')

        return np.array(self.uniques[y])


    def get_data_no_stop_words(self):

        X_train, y_train, X_test = self.get_data_no_numbers()

        # Use vectorizer here to convert to lower case, remove stop words
        # We don't actually perform bag of words here
        vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode", stop_words="english")
        analyser = vectorizer.build_analyzer()

        # Remvoe stop words
        for i in range(0, X_train.shape[0]):
            X_train[i] = " ".join(analyser(X_train[i]))

        for i in range(0, X_test.shape[0]):
            X_test[i] = " ".join(analyser(X_test[i]))

        # TODO: Lots of NLP
        # nlp = spacy.load('en')
        # -validate effectiveness of removing stop words and other transforms like number removal

        return X_train, y_train, X_test

    def get_bagofwords(self, max_features=5000):

        X_train, y_train, X_test = self.get_data_no_numbers()

        # Run bag of words, include basic stop words list, lower case transform
        vectorizer = CountVectorizer(analyzer= "word", strip_accents= "unicode", stop_words= "english", max_features=max_features)

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.fit_transform(X_test)

        y_train = self.convert_string_category_to_num(y_train)

        return X_train, y_train, X_test



