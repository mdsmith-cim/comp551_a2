from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Based on sklearn examples, specifically
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
# and http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py

# This svm approach uses nested cross validation to avoid having the validation set leak into the results
def svm(X_train, y_train, test_size=0.3, parameters=None):

    if parameters is None:
        parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': ['auto'],
                      'C': [0.1, 1, 10, 100, 1000, 10000]}

    # X train, X valid, y train, y valid
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=test_size)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=4, n_jobs=-1, verbose = 10)
    clf.fit(X_t, y_t)

    print("Best parameters set found on development set:\n")

    print(clf.best_params_)

    print("\nGrid scores on development set:\n")

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("\nDetailed classification report:\n")

    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_v, clf.predict(X_v)
    print(classification_report(y_true, y_pred))
    print()

    return clf