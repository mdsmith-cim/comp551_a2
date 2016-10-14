import numpy as np
from util.preprocess import preprocess
from sk_impl import grid_search_cv
from util.export import export
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Set path to data here if desired
pp = preprocess()

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

# Best parameters Linear:
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
