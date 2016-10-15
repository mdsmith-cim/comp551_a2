__author__ = 'Roman'

import csv
import math
from nltk import *
from math import e
from util import preprocess as b

#this function loads data from the kaggle files. It is not used in the final version of the code.
def load_data(filename):
    bob_the_processor = b.preprocess()
    bob,bob1,bob2 = bob_the_processor.get_data_nlp()
    #print(bob)
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    dataset.remove(dataset[0])
    print(dataset[0])
    print(dataset[1])
    return (dataset)

#converts a text file into a list of strings
def get_wordlist(filename):
    aaaa=csv.reader(open(filename, "r"))
    words = []
    for x in aaaa:
        if len(x)>0:
            this_line=str(x[0])
            bob = this_line.split(" ")
            words.append(bob[0])
    return words

#separates both train_out and train_in in 4 sets for cross-validation
def separateDataForCrossValidation (dataset1, dataset2,foldID ):
    size = 0.25 * len(dataset1)
    test_set1 = []
    test_set2 = []
    test_set1b = []
    test_set2b = []
    test_set1c = []
    test_set2c = []
    test_set1d = []
    test_set2d = []
    training_set1 = list(dataset1)
    training_set2 = list(dataset2)
    # a random datapoint is taken in the set, removed from the training set and added to test set
    #this is repeated until test set size is satisfied.
    i=0
    while len(test_set1) < size:
        test_set1.append(training_set1.pop(i))
        test_set2.append(training_set2.pop(i))
        test_set1b.append(training_set1.pop(i))
        test_set2b.append(training_set2.pop(i))
        test_set1c.append(training_set1.pop(i))
        test_set2c.append(training_set2.pop(i))
        test_set1d.append(training_set1.pop(i))
        test_set2d.append(training_set2.pop(i))

    final_test_set1 = []
    final_test_set2 = []
    if foldID == 0 :
        training_set1= test_set1b + test_set1c + test_set1d
        training_set2= test_set2b + test_set2c + test_set2d
        final_test_set1 = test_set1
        final_test_set2 = test_set2
    elif foldID ==1:
        training_set1= test_set1 + test_set1c + test_set1d
        training_set2= test_set2 + test_set2c + test_set2d
        final_test_set1 = test_set1b
        final_test_set2 = test_set2b
    elif foldID ==2:
        training_set1= test_set1 + test_set1b + test_set1d
        training_set2= test_set2 + test_set2b + test_set2d
        final_test_set1 = test_set1c
        final_test_set2 = test_set2c
    elif foldID ==3:
        training_set1= test_set1 + test_set1b + test_set1c
        training_set2= test_set2 + test_set2b + test_set2c
        final_test_set1 = test_set1d
        final_test_set2 = test_set2d
        # for x in training_set:
        #     test_set.append(training_set.pop(x))
    return training_set1, training_set2, final_test_set1, final_test_set2

#converts a string to a list of words.
#initially also did some data cleaning, and can be used to change the length of words considered (as discussed in reports)
def tokenize(string,nofly):
    #print (1)
    final_tokens = []
    tokens = string.split(" ")
    for i in tokens:
         if len(i) < 1:
            tokens.remove(i)
         elif '$' in i or '\\' in i:
             tokens.remove(i)
         elif i in nofly:
             tokens.remove(i)
         else:
             j = list(i)
             for k in j:
                 if k in [',', '.', ')', '(', ':', ';','_','{','}','[','[']:
                     j.remove(k)
             final_tokens.append("".join(j))
    return final_tokens

#returns classes from the input dataset. Nofly is the list of common words, it's in there because this function calls tokenize.
#this also computes the arrays of word frequencies in every class.
def get_classes(dataset, train_out,nofly):
    classes = {}
    #print(train_out[0])
    for i in range(len(train_out)):
        #tells user how far the program is in the task due to long computation times
        if i%10000==0:
            print (i)
        #computes word frequency in each class
        tokens = tokenize(dataset[i][1],nofly)
        if train_out[i][1] in classes.keys():
            for j in tokens:
                if (j) in classes[train_out[i][1]].keys():
                    classes[train_out[i][1]][j] += 1
                else:
                    classes[train_out[i][1]].update({j: 1})
        else:
            classes[train_out[i][1]] = {}
    return classes

#Computes P(Y), or the frequency of a class
def get_py(train_out, classes):
    classprobs = {}
    for x in classes.keys():
        classprobs[x] = 0
    tot = 0
    for i in train_out:
        if i[1] in classprobs.keys():
            classprobs[i[1]] += 1
            tot += 1
    for j in classprobs.keys():
        k = float(classprobs[j]) / float(tot)
        classprobs[j] = k
    return classprobs

#computes the probability of a class given a word is found.
def computeDiscreteProb(word, classes, classID):
    tot = 0
    #for each class, retrieves the frequency of the word.
    for c in classes.keys():
        if word in classes[c].keys():
            tot += classes[c][word]
        else:
            tot += 0
    #computes probability with some Laplace smoothing (
    if word in classes[classID].keys():
        prob = float(classes[classID][word]+1) / float(tot+4)
    else:
        prob = 1 / float(tot + 4)
    #returns probability and the total number of occurrences
    return prob,tot

#compute log-likelihood of a class given a sentence.
def computeClassLikelihood(input, classes, p_ys,nofly):
    tokens = tokenize(input,nofly)
    loglik = {}
    for c in classes.keys():
        #initialize log likelihood with log of P(Y) for a given class
        loglik.update({c: math.log(p_ys[c], e)})
        #then add the log of the probability of each token given class
        for i in tokens:
            probab,tot = computeDiscreteProb(i, classes, c)
            # print (probab)
            if tot>0:
                loglik[c] += math.log(probab, e)
    #then find the highest likelihood.
    maxlik = -10000
    bestclass = ""
    for classID in loglik:
        if loglik[classID] >= maxlik:
            maxlik = loglik[classID]
            bestclass = classID
    return bestclass

#generate predictions by calling computeClassLikelihood
def getPredictions(testSet, classes, p_ys,nofly):
    predictions = []
    for i in range(len(testSet)):
        if (i%10000==0):
            print(i)
        predictions.append([testSet[i][0], computeClassLikelihood(testSet[i][1], classes, p_ys,nofly)])
    return predictions

#tests those predictions
def evaluate(predictions, test_out):
    tot = 0
    n = 0
    for i in range(len(predictions)):
        #if there is a discrepancy between arrays, thee has to be a bug somewhere
        if str(predictions[i][0]) != str(test_out[i][0]):
            print("ALERT")
        if str(predictions[i][1]) == str(test_out[i][1]):
            n += 1
            tot += 1
        else:
            tot += 1
    print("prediction success", str(n), str(tot), str(float(n) / float(tot)))
    return (float(n) / float(tot))

#converts a string (sentence) to a list of words
def string_to_list(dataset):
    list = []
    for i in range(len(dataset)):
        list.append([i,dataset[i]])
    return list


###THIS IS AN EXAMPLE OF HOW THE PROGRAM WOULD BE RUN WITHOUT CROSS VALIDATION###
#dataset = load_data("train_in.csv")
#bob_the_processor = b.preprocess()
#common_words = get_wordlist("1-1000.txt")
#dataset,train_out,final_test = bob_the_processor.get_data_nlp()
#dataset = string_to_list(dataset)
#train_out=string_to_list(train_out)
#final_test=string_to_list(final_test)
#print(dataset)
#train_out = load_data("train_out.csv")
#final_test = load_data("test_in.csv")
#trainingSet, train_out, testSet, test_out = separateDataForCrossValidation(dataset, train_out,0)
#print (trainingSet[0])
#print (train_out[0])
#classes = get_classes(trainingSet, train_out, common_words)
#p_ys = get_py(train_out, classes)
#print(p_ys)
# input = ['170', 'In this article a stabilizing feedback control is computed for a semilinear parabolic partial differential equation utilizing a nonlinear model predictive (NMPC) method. In each level of the NMPC algorithm the finite time horizon open loop problem is solved by a reduced-order strategy based on proper orthogonal decomposition (POD). A stability analysis is derived for the combined POD-NMPC algorithm so that the lengths of the finite time horizons are chosen in order to ensure the asymptotic stability of the computed feedback controls. The proposed method is successfully tested by numerical examples.']
#bestClass = computeClassLikelihood(input,classes,p_ys)
#predictions = getPredictions(testSet,classes,p_ys,common_words)
#evaluate(predictions,test_out)
#predictions = getPredictions(trainingSet, classes, p_ys,common_words)
#evaluate(predictions, train_out)
#pred_fin = getPredictions(final_test,classes,p_ys,common_words)
#with open ("submit.txt",'w') as file:
 #   file.write("id,category\n")
  #  cats = ['math', 'cs', 'stat', 'physics']
   # for i in pred_fin:
    #    file.write(str(i[0]) + "," + cats[i[1]]+"\n" )

