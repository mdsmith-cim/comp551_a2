import numpy as np
#from numpy import linalg as LA
import random
from util.preprocess import preprocess
#from scipy import sparse
#import time
from scipy.sparse.linalg import norm
import math
from scipy.sparse import csr_matrix
from statistics import mode, StatisticsError, stdev, mean
#import random
#from sklearn.model_selection import train_test_split


# not used for cross validation
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
def evaluate(predictions, test_out, descr):
    tot = 0
    n = 0
    
    for key in predictions:
        if predictions[key] == test_out[key]:
            n += 1
            tot += 1
        else:
            tot += 1
    print(descr, 'prediction success', n, tot, n/tot)
    accuracy = n/tot
    return accuracy


# returns the prediction based on a majority rule
def pick_index(k, thelist, index_list, metric, y_train):
# k is for kNN
# thelist is not important for majority rule but for picking min, yes
# index_list is the list of indices which are closest to the validation example
    
    class_list = []
    for i in range(k):
        class_list.append(y_train[index_list[i]])
    
    try: # there is a unique mode
        md = mode(class_list)
#        print('mode',md)
        pred = md
    except StatisticsError: # mode not unique
        
#        print("There is no mode!")
        
        # pick max in thelist
        if(metric == 'dot'):
            max_dot = max(thelist)
            max_dot_index = thelist.index(max_dot)
            index = index_list[max_dot_index]
            pred = y_train[index]
        
        if(metric == 'euc'):
            min_euc = min(thelist)
            min_euc_index = thelist.index(min_euc)
            index = index_list[min_euc_index]
            pred = y_train[index]
            
    return pred



''' COULD TRY DOT PRODUCT AGAIN '''



def get_predictions(X_train, x_test, train_ind, valid_ind, k, y_train):
    
    X_copy = X_train
    X_train = X_train[train_ind]
    y_train = y_train[train_ind]
    
    diff = []
    y_pred = dict()
    
    count = 1    
       
    for v_ind in valid_ind:
        
        # validation example is from training set, index v_ind
        test = X_copy[v_ind]
        diff = csr_matrix(X_train.toarray() - test)
        norms = norm(diff,ord=1,axis=1) # find norm of each row
        
        # norms should no longer be sparse. make list out of it
        norms = norms.tolist()

        # will make changes to this one and want to keep copy of norms
        temp_norms = norms
        
        
        # kNN
        
        k_neigh = []
        neigh_norms = []
        
        for i in range(k):

            # find min norm
            min_euc = min(temp_norms)
            # add to list
            neigh_norms.append(min_euc)
            # find index of min norm
            min_euc_index = temp_norms.index(min_euc)
            # add to list
            k_neigh.append(min_euc_index)
            # remove from temp_norms so get new min next time if k>1
            del temp_norms[min_euc_index]
        
        # get prediction
        pred = pick_index(k, neigh_norms, k_neigh, 'euc', y_train)
        
        y_pred[v_ind] = pred
        
        count += 1
        
#        if count == 10022:
#            break
    return y_pred







''' ACTUALLY RUNNING THINGS STARTING HERE '''

k = 1


# Set path to data here if desired
pp = preprocess()

# X_train is with features as words
# y_train is classification of X_train
# X_test is test data with features as words
X_train, y_train, X_test = pp.get_tf_idf(max_features=100, use_spacy=True)

X_train = X_train[0:2216*16]
y_train = y_train[0:2216*16]

# total number of training examples
num_tot_train = X_train.shape[0]
# ratio of training examples to be used for validation
#valid_ratio = 1/4

# separate training set into actual training and validation
#train_ind, valid_ind = data_separation(num_tot_train,valid_ratio)


# get the predictions
#y_pred = get_predictions(X_train, X_train, train_ind, valid_ind, k, y_train)

# print out accuracy
#accuracy = evaluate(y_pred, y_train,'precrossval')

#all training examples and 100 features
#prediction success 7598 8864 0.8571750902527075

    
    
def crossval_sep(num_tot_train, ratio):
    # size of validation set
    size = math.ceil(ratio*num_tot_train)
    
    # all data
    all_indices = list(range(0,num_tot_train))
    random.shuffle(all_indices)
    
    # split into three lists
    list1 = all_indices[0:size]
    list2 = all_indices[size:2*size]
    list3 = all_indices[2*size:3*size]
    list4 = all_indices[3*size:]
    
    # three folds of cross validation
    train_ind1 = list1+list2+list3
    valid_ind1 = list4
   
    train_ind2 = list1+list2+list4
    valid_ind2 = list3
    
    train_ind3 = list1+list3+list4
    valid_ind3 = list2
    
    train_ind4 = list2+list3+list4
    valid_ind4 = list1
    
    return train_ind1, valid_ind1, train_ind2, valid_ind2, train_ind3, valid_ind3, train_ind4, valid_ind4



''' CROSS VALIDATION BEGINS '''

k_options = [1,3,4,8]
#k_options = []
valid_ratio = 1/4

train_ind1, valid_ind1, train_ind2, valid_ind2, train_ind3, valid_ind3, train_ind4, valid_ind4 = crossval_sep(num_tot_train,valid_ratio)



''' TRAINING ERROR '''
train_ind = range(num_tot_train)
#print(train_ind)
valid_ind = range(num_tot_train)
#print(valid_ind)


# loop over k after finding training and validation sets.

# for each k option
#    for each fold of cross validation
#        validate on kth subset.. find error.
#    find average prediction error over k folds
# choose k with lowest average prediction error. or equivalently, highest accuracy.

avg_acc = dict()
stdev_acc = dict()

train_acc = dict()

for k in k_options:
    print('k =',k)
    
    # validation
    y_pred1 = get_predictions(X_train, X_train, train_ind1, valid_ind1, k, y_train)
    acc1 = evaluate(y_pred1, y_train, 'fold1')
    
    y_pred2 = get_predictions(X_train, X_train, train_ind2, valid_ind2, k, y_train)
    acc2 = evaluate(y_pred2, y_train, 'fold2')
    
    y_pred3 = get_predictions(X_train, X_train, train_ind3, valid_ind3, k, y_train)
    acc3 = evaluate(y_pred3, y_train, 'fold3')
    
    y_pred4 = get_predictions(X_train, X_train, train_ind4, valid_ind4, k, y_train)
    acc4 = evaluate(y_pred4, y_train, 'fold4')
    
    avg_acc[k] = mean([acc1,acc2,acc3,acc4])
    stdev_acc[k] = stdev([acc1,acc2,acc3,acc4])    
    
    # training
    y_pred_train = get_predictions(X_train, X_train, train_ind, valid_ind, k, y_train)
    train_acc[k] = evaluate(y_pred_train, y_train, 'training')
    

print('')
print('average accuracy for each k',avg_acc)
print('stdev accuracy for each k',stdev_acc)
print('training accuracy',train_acc)

#optimal_k = max(avg_acc, key=avg_acc.get)

print('\n')





