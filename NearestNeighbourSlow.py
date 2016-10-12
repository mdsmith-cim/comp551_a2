import numpy as np
from numpy import linalg as LA
import random
from util.preprocess import preprocess
from scipy import sparse
import time
from scipy.sparse.linalg import norm
import math
from scipy.sparse import csr_matrix



# separate training data into training and validation randomly
def data_separation(num_tot_train, ratio):
    
    # size of validation set
    size = math.ceil(ratio*num_tot_train)
    
    # random validation indices
    validation_indices = np.random.choice(num_tot_train,size,replace=False)
    
    # remove validation indices from training indices
    training_indices = list(range(0,num_tot_train))
    training_indices = np.delete(training_indices,validation_indices).tolist()
        
    return training_indices, validation_indices


# finds the accuracy
def evaluate(predictions, test_out):
    tot = 0
    n = 0
    
    for key in predictions:
        if predictions[key] == test_out[key]:
            n += 1
            tot += 1
        else:
            tot += 1
    print('prediction success', n, tot, n/tot)



def get_predictions(X_train, x_test, train_ind, valid_ind):
    
    # pick metric
    dotprod = 1 # not really a metric but works better
    euclid = 1
    
    diff = []
    y_pred_dot = dict()
    y_pred_euc = dict()
    
    count = 1    
       
    for v_ind in valid_ind:
        
        # validation example is from training set, index v_ind
        test = X_train[v_ind]
        
        dot = 0
        diff = norm(test - X_train[train_ind[0]])
        index_dot = 0
        index_euc = 0
        
        # compare test to all examples in training set
        for t_ind in train_ind:
            
            if dotprod == 1:
                new_dot = test.dot(X_train[t_ind].T).toarray()

                if(new_dot > dot):
                    index_dot = t_ind
                    dot = new_dot                
                
                
            elif euclid == 1:
                new_diff = norm(test - X_train[t_ind])

                if(new_diff < diff):
                    index_euc = t_ind
                    diff = new_diff
            
        y_pred_dot[v_ind] = y_train[index_dot]
        y_pred_euc[v_ind] = y_train[index_euc]        
        
        count += 1
        print(count)
        
        if count == 10:
            break    

    return y_pred_dot, y_pred_euc







''' ACTUALLY RUNNING THINGS STARTING HERE '''

# Set path to data here if desired
pp = preprocess()

# X_train is with features as words
# y_train is classification of X_train
# X_test is test data with features as words
#X_train, y_train, X_test = pp.get_bagofwords() #changing max features does not speed things up much.
X_train, y_train, X_test = pp.get_tf_idf(max_features=10000, use_spacy=True)



# total number of training examples
num_tot_train = X_train.shape[0]
# ratio of training examples to be used for validation
valid_ratio = 1/10

# separate training set into actual training and validation
train_ind, valid_ind = data_separation(num_tot_train,valid_ratio)
num_train = len(train_ind)
num_valid = len(valid_ind)


# get the predictions
y_pred_dot, y_pred_euc = get_predictions(X_train, X_train, train_ind, valid_ind)

# print out accuracy
evaluate(y_pred_dot, y_train)
evaluate(y_pred_euc, y_train)
   


'''
cross validation to pick the best k

problems
too many zeros. picks abstract with least number of words.
maybe to combat this take bunch of nearest neighbours.. but then will always take shortest ones?

'''
'''
TO DO
deloop get_predictions
do some cross validation

'''
