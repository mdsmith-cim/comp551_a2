import numpy as np
from util.preprocess import preprocess
from sk_impl import grid_search_cv
from util.export import export
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
X_train, y_train, X_test = pp.get_tf_idf(max_features=10000, use_spacy=True)


# Remove pp object to save memory
# having spacy in memory consumes ~1.7GB

# Do stuff with data...

classifierLin = grid_search_cv(X_train, y_train, parameters={'C': [0.01, 0.1, 1, 10, 100],
                                                             'loss': ['hinge', 'squared_hinge'],
                                                             'tol': [1e-6, 1e-5, 1e-4, 1e-3]},
                               classifier=LinearSVC(verbose=10, class_weight='balanced', max_iter=10000))

classifierRandF = grid_search_cv(X_train, y_train, parameters={'n_estimators': [5, 10, 15, 20, 25, 30],
                                                               'criterion': ['gini', 'entropy']},
                                 classifier=RandomForestClassifier(n_jobs=-1, verbose=10,
                                                                   class_weight='balanced'))

prediction = classifierLin.predict(X_test)

export(pp.convert_num_category_to_string(prediction), 'linearSVM.csv')
