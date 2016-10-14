import numpy as np
from util.preprocess import preprocess
from sk_impl import grid_search_cv
from util.export import export
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Set path to data here if desired
pp = preprocess()

# Raw data
#X_train, y_train, X_test = pp.get_data_raw()

# No numbers, but otherwise identical to raw including string categories
#X_train, y_train, X_test = pp.get_data_no_numbers()

# With no numbers and stop words removed; basically just words with spaces in between
#X_train, y_train, X_test = pp.get_data_no_stop_words()

# All of the above + processed into bag of words
# Output is a sparse matrix
# Set the max_features argument if you want to change the limit at which it cuts off (default: 5000)
# Bag of words will provide the MAX_FEATURES most common words
# Even with 5000 features, one abstract might only have 70-100 non-zero features (words)
# Execution time is ~25 sec on my laptop
#X_train, y_train, X_test = pp.get_bagofwords(max_features=5000, use_spacy=False)

# If you want bag of words based on the root of words from spacy natural language processing
#X_train, y_train, X_test = pp.get_bagofwords(max_features=5000, use_spacy=True)


# If you want data processed with spacy but not yet run through bag of words:
#X_train, y_train, X_test = pp.get_data_nlp()

# tf-idf is available
# Note: all arguments are passed directly to get_bagofwords
#X_train, y_train, X_test = pp.get_tf_idf(max_features=10000, use_spacy=True)


X_train, y_train, X_test = pp.get_tf_idf(max_features=20000, use_spacy=True, ngram_range=(1, 2))

# Remove pp object to save memory
# having spacy in memory consumes ~1.7GB

# Do stuff with data...

classifierLin = grid_search_cv(X_train, y_train, parameters={'C': [0.01, 0.1, 1, 10, 100],
                                                             'loss': ['hinge', 'squared_hinge'],
                                                             'tol': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                                                             'class_weight': ['balanced', None],
                                                             'fit_intercept': [True, False],
                                                             'penalty': ['l2'],
                                                             'dual': [True, False]},
                               classifier=LinearSVC(verbose=10, max_iter=100000), error_score=0)

# Best parameters:
# {'C': 1,
#  'class_weight': None,
#  'dual': True,
#  'fit_intercept': True,
#  'loss': 'hinge',
#  'penalty': 'l2',
#  'tol': 1e-09}

classifierRandF = grid_search_cv(X_train, y_train, parameters={'n_estimators': [5, 10, 15, 20, 25, 30],
                                                               'criterion': ['gini', 'entropy']},
                                 classifier=RandomForestClassifier(n_jobs=-1, verbose=10,
                                                                   class_weight='balanced'))

classifierSVM = grid_search_cv(X_train, y_train, parameters={'kernel': ['rbf'],
                                                              'C' : [0.1, 0.5, 1],
                                                              'gamma': [0.1, 0.5, 1]},
                               classifier=SVC(cache_size=4096, verbose=True,
                                              decision_function_shape='ovr', max_iter=500000))


# Now that we have appropriate parameters after a few hours of cross validation, train on all available data


# Refit to all train data
finalClass = LinearSVC(verbose=10, max_iter=100000, **classifierLin.best_params_)
finalClass.fit(X_train,y_train)

prediction = finalClass.predict(X_test)

export(pp.convert_num_category_to_string(prediction), 'linearSVM.csv')
