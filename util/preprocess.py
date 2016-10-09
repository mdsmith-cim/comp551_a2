import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import numpy as np
import spacy
import re
import os


class preprocess:

    def __init__(self, train_in = 'data/train_in.csv', train_out = 'data/train_out.csv', test_in = 'data/test_in.csv'):

        self.categories = ['math', 'cs', 'stat', 'physics']
        self.uniques = None
        self.train_in = train_in
        self.train_out = train_out
        self.test_in = test_in

        # Below is for NLP processing only
        self.STOP_WORDS_PLUS_EXTRA = frozenset(list(ENGLISH_STOP_WORDS) + stopwords.words("english") + [';', "'s"])
        self.SYMBOL_REMOVE_LIST = frozenset(["(", ")", "^", ".", ",", "[", "]", "-", "--", "{", "}", "=", "/", ":",
                                             "`", "|", "<", ">", "\\", '"', "'", "%", "*"])
        self.nlp = spacy.load('en')

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

        # Remove stop words
        for i in range(0, X_train.shape[0]):
            X_train[i] = " ".join(analyser(X_train[i]))

        for i in range(0, X_test.shape[0]):
            X_test[i] = " ".join(analyser(X_test[i]))

        return X_train, y_train, X_test

    def get_bagofwords(self, max_features=5000, data_directory='nlp_processed/', clean=False, use_disk=True, use_spacy=True):

        filename = os.path.join(data_directory, 'data_bow.dat')

        if clean:
            os.remove(filename)
            print("Deleted existing data from disk")

        if use_disk:
            try:
                file = open(filename, 'rb')
                loaded = np.load(file)

                if 'X_train' not in loaded.files or 'X_test' not in loaded.files or 'y_train' not in loaded.files:
                    file.close()
                    raise Exception('Saved data is corrupted')

                X_train = loaded['X_train']
                X_test = loaded['X_test']
                y_train = loaded['y_train']

                file.close()

                print("Loaded data from file " + filename)

                return X_train, y_train, X_test

            except Exception as e:
                print("Error loading saved data;" + str(e))

                try:
                    os.mkdir(data_directory)
                # do nothing if it already exists
                except FileExistsError:
                    pass

        if use_spacy:

            X_train, y_train, X_test = self.get_data_raw()
            # Bag of words: run on preprocessed data
            vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode", tokenizer=self.process_spacy_sample,
                                         max_features=max_features)
        else:
            X_train, y_train, X_test = self.get_data_no_numbers()
            # Bag of words: use scikit-learn built in stop words
            vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode", stop_words="english",
                                         max_features=max_features)

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.fit_transform(X_test)

        y_train = self.convert_string_category_to_num(y_train)

        if use_disk:
            file = open(filename, 'wb')
            np.savez_compressed(file, X_train=X_train, X_test=X_test, y_train=y_train)
            print("Saved data to " + filename)
            file.close()

        return X_train, y_train, X_test

    def get_data_nlp(self, use_disk=True, data_directory='nlp_processed/', clean=False):

        filename = os.path.join(data_directory, 'data_nlp.dat')

        if clean:
            os.remove(filename)
            print("Deleted existing data from disk")

        if use_disk:

            try:
                file = open(filename, 'rb')
                loaded = np.load(file)

                if 'X_train' not in loaded.files or 'X_test' not in loaded.files or 'y_train' not in loaded.files:
                    file.close()
                    raise Exception('Saved data is corrupted')

                X_train = loaded['X_train']
                X_test = loaded['X_test']
                y_train = loaded['y_train']

                file.close()

                print("Loaded data from file " + filename)

                return X_train, y_train, X_test

            except Exception as e:
                print("Error loading saved data;" + str(e))

                try:
                    os.mkdir(data_directory)
                # do nothing if it already exists
                except FileExistsError:
                    pass

        X_train, y_train, X_test = self.get_data_raw()

        # If you dont' have this, follow this guide
        # https://spacy.io/docs/#getting-started

        X_train = np.array(self.process_spacy(X_train))
        X_test = np.array(self.process_spacy(X_test))

        y_train = self.convert_string_category_to_num(y_train)

        if use_disk:
            file = open(filename, 'wb')
            np.savez_compressed(file, X_train=X_train, X_test=X_test, y_train=y_train)
            print("Saved data to " + filename)
            file.close()

        return X_train, y_train, X_test

    def process_spacy(self, data):

        results = []

        for i in range(0, data.shape[0]):

            results.append(" ".join(self.process_spacy_sample(data[i])))

        return results

    def process_spacy_sample(self, text):

        #  We need to do a little preprocessing to remove some symbols otherwise spacy does strange things when
        # it parses

        # Remove any newlines just in case (don't think there are any in the data but just in case)
        text = text.strip()

        # Strip latex - cheap method: remove everything between $'s
        text = re.sub('\$[^$]+\$', '', text)

        for rm in self.SYMBOL_REMOVE_LIST:
            text = text.replace(rm, " ")

        lemmas = []
        parsed = self.nlp(text)
        for token in parsed:
            # token has lemma unless it is a pronoun, in which case it is assigned the string -PRON-
            # so in that case we simply use the word itself
            # See https://spacy.io/docs/#token
            if not token.is_digit:
                lemmas.append(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)

        parsed = lemmas

        # Remove all stop words and other undesirables
        # also make sure not an empty string
        parsed = [token for token in parsed if
                  (token not in self.STOP_WORDS_PLUS_EXTRA and len(token) != 0)]

        return parsed

