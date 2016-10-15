How to run this code :


Text processing : run util/preprocess.py, create a preprocess instance and call a function in preprocess (example in naive_bayes_cross_validation.py)
Naive Bayes : To run cross validation on Naive Bayes with the best set of features, run naive_bayes_cross_validation.py. This script calls final_naive_bayes.py, which contains the actual naive bayes code
SVMs : run main.py. If a function is not included in main.py, it can be found in old_examples.py

kNN:
Run kNN.py To change the norm:
2-norm (default)
Line 99: norms = norm(diff,axis=1)
1-norm
Line 99: norms = norm(diff,ord=1,axis=1)
inf-norm
Line 99: norms = norm(diff,ord=inf,axis=1)