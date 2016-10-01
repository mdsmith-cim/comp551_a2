import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
# If you dont' have this, follow this guide
# https://spacy.io/docs/#getting-started
#import spacy

class preprocess:

    def __init__(self, process):

        self.process = process

    def get_data_raw(self, train_in = 'data/train_in.csv', train_out = 'data/train_out.csv', test_in = 'data/test_in.csv'):

        train = pd. read_csv(train_in)
        train_y = pd.read_csv(train_out)
        test = pd.read_csv(test_in)

        X_train = train['abstract'].values
        X_test = test['abstract'].values
        y_train = train_y['category'].values

        return X_train, y_train, X_test

    def get_data_process(self, max_features=5000):

        X_train, y_train, X_test = self.get_data_raw()

        # Remove anything not a letter
        for i in range(0, X_train.shape[0]):
            X_train[i] = re.sub("[^a-zA-Z]", " ", X_train[i])

        for i in range(0, X_test.shape[0]):
            X_test[i] = re.sub("[^a-zA-Z]", " ", X_test[i])

        # TODO: Lots of NLP
        # nlp = spacy.load('en')
        # -validate effectiveness of removing stop words and other transforms like number removal

        # Run bag of words, include basic stop words list, lowercase transform is default
        vectorizer = CountVectorizer(analyzer= "word", strip_accents= "unicode", stop_words= "english", max_features=max_features )

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.fit_transform(X_test)

        return X_train, y_train, X_test



