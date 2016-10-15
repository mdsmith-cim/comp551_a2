__author__ = 'Roman'
import final_naive_bayes as nb
import main
import util
import util.preprocess as b
import math
import copy
import statistics
val=[]
train = []

#opens an instance of preprocess
bob_the_processor = b.preprocess()
#gets a list of words from the list of common words
common_words = nb.get_wordlist("1-1000.txt")
#initiates arrays from the data
dataset,train_out,final_test = bob_the_processor.get_data_nlp()

#Copy the arrays for cross-validation.
dataset0 = nb.string_to_list(dataset)
train_out0=nb.string_to_list(train_out)
dataset1 = copy.copy(dataset0)
train_out1 = copy.copy(train_out0)
dataset2 = copy.copy(dataset0)
train_out2 = copy.copy(train_out0)
dataset3 = copy.copy(dataset0)
train_out3 = copy.copy(train_out0)


#get training and validation sets from the first copy of the array
trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset0, train_out0,0)
#identify classes
classes = nb.get_classes(trainingSet, train_out, common_words)
#count the proportions of classes.
p_ys = nb.get_py(train_out, classes)
print(p_ys)
#generates predictions for validation set
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
#appends predictions to the cross-validation array
val.append(nb.evaluate(predictions,test_out))
#generate predictions for training set
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
#append to cross-validation array
train.append(nb.evaluate(predictions, train_out))
del(bob_the_processor)



##repeat these steps three times to complete 4-fold cross-validation###
trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset1, train_out1,1)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))

trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset2, train_out2,2)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))

trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset3, train_out3,3)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))


##print output
print(val)
print(train)
print (statistics.mean(val))
print (statistics.mean(train))
print (statistics.stdev(val))
print (statistics.stdev(train))