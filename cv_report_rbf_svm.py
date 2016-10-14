import numpy as np
from util.preprocess import preprocess
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Set path to data here if desired
pp = preprocess()

X_train, y_train, X_test = pp.get_tf_idf(max_features=20000, use_spacy=True, ngram_range=(1, 2))

pa = {'C': 5, 'gamma': 1, 'kernel': 'rbf'}

svc = SVC(cache_size=4096, verbose=True, decision_function_shape='ovr', max_iter=500000, **pa)

kf = KFold(n_splits=4, shuffle=True)

trainScores = []
testScores = []

for train, test in kf.split(X_train):

    X_tr = X_train[train]
    X_t = X_train[test]

    y_tr = y_train[train]
    y_t = y_train[test]

    svc.fit(X_tr, y_tr)

    trainScores.append(svc.score(X_tr, y_tr))
    testScores.append(svc.score(X_t, y_t))

trainScores = np.array(trainScores)
testScores = np.array(testScores)

print()
print("Avg. train accuracy: %0.3f (std dev. %0.3f)" % (trainScores.mean(), trainScores.std()))
print("Avg. validation accuracy: %0.3f (std dev. %0.3f)" % (testScores.mean(), testScores.std()))
