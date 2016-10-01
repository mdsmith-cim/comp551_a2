__author__ = 'Roman'

import random
import csv
import nltk
import math
from math import e
import numpy as np
import scipy.stats as sp
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


def load_data(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    dataset.remove(dataset[0])
    print(dataset[0])
    print(dataset[1])
    return (dataset)


def get_wordlist(filename):
    aaaa=csv.reader(open(filename, "r"))
    words = []
    for x in aaaa:
        this_line=str(x[0])
        bob = this_line.split(" ")
        words.append(bob[1])
    # print(words[1].strip() + '\n' for words in line_words)
    return bob


def separateData(dataset1, dataset2, ratio):
    size = ratio * len(dataset1)
    test_set1 = []
    test_set2 = []
    training_set1 = list(dataset1)
    training_set2 = list(dataset2)
    # a random datapoint is taken in the set, removed from the training set and added to test set
    #this is repeated until test set size is satisfied.
    while len(test_set1) < size:
        datapoint = random.randrange(len(training_set1))
        test_set1.append(training_set1.pop(datapoint))
        test_set2.append(training_set2.pop(datapoint))
        # for x in training_set:
        #     test_set.append(training_set.pop(x))
    print(training_set1[0][0])
    print(training_set1[133])
    print(training_set1[1456][0])
    print(training_set1[1456][0])
    print(training_set1[14560][0])
    print(training_set2[22000][0])
    print(training_set2[133])
    print(training_set2[1456][0])
    print(training_set2[14560][0])
    print(training_set2[22000][0])
    return training_set1, training_set2, test_set1, test_set2


def tokenize(string,nofly):
    tokens = string.split(" ")
    for i in tokens:
        j = list(i)
        for k in j:
            if k in [',', '.', ')', '(', ':', ';']:
                j.remove(k)
        "".join(j)
        if len(j) < 4:
            tokens.remove(i)
        elif i in nofly:
            tokens.remove(i)

    return tokens


def get_classes(dataset, train_out,nofly):
    classes = {}
    print(train_out[0])
    for i in range(len(train_out)):
        tokens = tokenize(dataset[i][1],nofly)
        # print (tokens)
        # print (dataset[i][0])
        if train_out[i][1] in classes.keys():
            for j in tokens:
                #   print (j)
                if (j) in classes[train_out[i][1]].keys():
                    classes[train_out[i][1]][j] += 1
                    #print (classes[train_out[i][1]].keys())
                else:
                    classes[train_out[i][1]].update({j: 1})
                    #print (classes[train_out[i][1]].keys())

        else:
            classes[train_out[i][1]] = {}

            # print(classes["math"])
    return classes


def get_py(train_out, classes):
    classprobs = {}
    for x in classes.keys():
        classprobs[x] = 0
    tot = 0
    for i in train_out:
        classprobs[str(i[1])] += 1
        tot += 1
    for j in classprobs.keys():
        k = float(classprobs[j]) / float(tot)
        classprobs[j] = k

        # print (classprobs)
    return classprobs


def computeDiscreteProb(word, classes, classID):
    tot = 0
    for c in classes.keys():
        if word in classes[c].keys():
            tot += classes[c][word]
        else:
            tot += 0
    if word in classes[classID].keys():
        prob = float(classes[classID][word]) / float(tot)
    else:
        prob = 1 / float(tot + 1)
    # print (prob)
    return prob


def computeClassLikelihood(input, classes, p_ys,nofly):
    tokens = tokenize(input[1],nofly)
    loglik = {}
    for c in classes.keys():
        loglik.update({c: math.log(p_ys[c], e)})
        for i in tokens:
            probab = computeDiscreteProb(i, classes, c)
            # print (probab)
            loglik[c] += math.log(probab, e)
    maxlik = -1000
    bestclass = ""
    for classID in loglik:
        if loglik[classID] >= maxlik:
            maxlik = loglik[classID]
            bestclass = classID
    # print (loglik)
    return bestclass


def getPredictions(testSet, classes, p_ys,nofly):
    predictions = []
    for i in range(len(testSet)):
        predictions.append([testSet[i][0], computeClassLikelihood(testSet[i][1], classes, p_ys,nofly)])
    return predictions


def evaluate(predictions, test_out):
    tot = 0
    n = 0
    for i in range(len(predictions)):
        if str(predictions[i][0]) != str(test_out[i][0]):
            print("ALERT")
        if str(predictions[i][1]) == str(test_out[i][1]):
            n += 1
            tot += 1
        else:
            tot += 1
    print("prediction success", str(n), str(tot), str(float(n) / float(tot)))
    # for k in range(len(predictions)):
    # print(predictions[k][0],predictions[k][1],test_out[k][0],test_out[k][1])


dataset = load_data("train_in.csv")
common_words = get_wordlist("common_words.txt")
train_out = load_data("train_out.csv")
trainingSet, train_out, testSet, test_out = separateData(dataset, train_out, 0.1)
classes = get_classes(trainingSet, train_out, common_words)
p_ys = get_py(train_out, classes)
print(p_ys)
# input = ['170', 'In this article a stabilizing feedback control is computed for a semilinear parabolic partial differential equation utilizing a nonlinear model predictive (NMPC) method. In each level of the NMPC algorithm the finite time horizon open loop problem is solved by a reduced-order strategy based on proper orthogonal decomposition (POD). A stability analysis is derived for the combined POD-NMPC algorithm so that the lengths of the finite time horizons are chosen in order to ensure the asymptotic stability of the computed feedback controls. The proposed method is successfully tested by numerical examples.']
#bestClass = computeClassLikelihood(input,classes,p_ys)
#predictions = getPredictions(testSet,classes,p_ys)
#evaluate(predictions,test_out)
predictions = getPredictions(trainingSet, classes, p_ys,common_words)
evaluate(predictions, train_out)
