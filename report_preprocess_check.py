import numpy as np
from util.preprocess import preprocess
from sk_impl import grid_search_cv
from util.export import export
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Set path to data here if desired
pp = preprocess()

clf = LinearSVC(max_iter=100000, C=1, class_weight=None, dual=True, fit_intercept=True, loss='hinge',
                penalty='l2', tol=1e-09)

# ------

print("Baseline: 5000 features, using spacy, ngram = (1,1)")

X_train, y_train, X_test = pp.get_tf_idf(max_features=5000, use_spacy=True, ngram_range=(1, 1))

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with baseline: {0}".format(clf.score(X_v, y_v)))

# -----------

print("Testing ngram = (1,2)")

X_train, y_train, X_test = pp.get_tf_idf(max_features=5000, use_spacy=True, ngram_range=(1, 2), data_directory='ngram/')

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with ngram = (1,2): {0}".format(clf.score(X_v, y_v)))

# -----

print("Testing without spacy; basic stopwords, strip all symbols (including 's , 't i.e. don't etc.), no word roots")

X_train, y_train, X_test = pp.get_tf_idf(max_features=5000, use_spacy=False, ngram_range=(1, 1), data_directory='no_spacy/')

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score without spacy: {0}".format(clf.score(X_v, y_v)))

# -------

print("Get max number of features - no processing at all")

X_train, y_train, X_test = pp.get_data_raw()
vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode")
print("# features: {0}".format(vectorizer.fit_transform(X_train).shape[1]))

print("Get max number of features - basic scikit stop words removal")

vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode", stop_words="english")
print("# features: {0}".format(vectorizer.fit_transform(X_train).shape[1]))

print("Get max number of features - basic scikit + strip all non a-z [basically no spacy]")
X_train, y_train, X_test = pp.get_data_no_numbers()
vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode", stop_words="english")
print("# features: {0}".format(vectorizer.fit_transform(X_train).shape[1]))

print("Get max number of features - spacy")
X_train, y_train, X_test = pp.get_data_raw()
vectorizer = CountVectorizer(analyzer="word", strip_accents="unicode", tokenizer=pp.process_spacy_sample)
print("# features: {0}".format(vectorizer.fit_transform(X_train).shape[1]))

print("Bag of words instead of tf-idf")

X_train, y_train, X_test = pp.get_bagofwords(max_features=5000, use_spacy=True, ngram_range=(1, 1))

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with BOW: {0}".format(clf.score(X_v, y_v)))

# ------


print("Testing with 1000 features")

X_train, y_train, X_test = pp.get_tf_idf(max_features=1000, use_spacy=True, ngram_range=(1, 1), data_directory='max_feat_1000/')

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with 1000 features: {0}".format(clf.score(X_v, y_v)))

# ----


print("Testing with 10000 features")

X_train, y_train, X_test = pp.get_tf_idf(max_features=10000, use_spacy=True, ngram_range=(1, 1), data_directory='max_feat_10000/')

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with 10000 features: {0}".format(clf.score(X_v, y_v)))


# ------

print("Testing with 20000 features")

X_train, y_train, X_test = pp.get_tf_idf(max_features=20000, use_spacy=True, ngram_range=(1, 1), data_directory='max_feat_20000/')

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with 20000 features: {0}".format(clf.score(X_v, y_v)))

# ------

print("Testing with 40000 features")


X_train, y_train, X_test = pp.get_tf_idf(max_features=20000, use_spacy=True, ngram_range=(1, 1), data_directory='max_feat_40000/')

X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

clf.fit(X_t, y_t)

print("Score with 40000 features: {0}".format(clf.score(X_v, y_v)))