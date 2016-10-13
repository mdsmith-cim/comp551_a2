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

bob_the_processor = b.preprocess()
common_words = nb.get_wordlist("1-1000.txt")
dataset,train_out,final_test = bob_the_processor.get_data_no_stop_words()
dataset0 = nb.string_to_list(dataset)
train_out0=nb.string_to_list(train_out)
dataset1 = copy.copy(dataset0)
train_out1 = copy.copy(train_out0)
dataset2 = copy.copy(dataset0)
train_out2 = copy.copy(train_out0)
dataset3 = copy.copy(dataset0)
train_out3 = copy.copy(train_out0)
final_test0=nb.string_to_list(final_test)
#train_out = load_data("train_out.csv")
#final_test = load_data("test_in.csv")
trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset0, train_out0,0)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
# input = ['170', 'In this article a stabilizing feedback control is computed for a semilinear parabolic partial differential equation utilizing a nonlinear model predictive (NMPC) method. In each level of the NMPC algorithm the finite time horizon open loop problem is solved by a reduced-order strategy based on proper orthogonal decomposition (POD). A stability analysis is derived for the combined POD-NMPC algorithm so that the lengths of the finite time horizons are chosen in order to ensure the asymptotic stability of the computed feedback controls. The proposed method is successfully tested by numerical examples.']
#bestClass = computeClassLikelihood(input,classes,p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))
del(bob_the_processor)


trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset1, train_out1,1)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
# input = ['170', 'In this article a stabilizing feedback control is computed for a semilinear parabolic partial differential equation utilizing a nonlinear model predictive (NMPC) method. In each level of the NMPC algorithm the finite time horizon open loop problem is solved by a reduced-order strategy based on proper orthogonal decomposition (POD). A stability analysis is derived for the combined POD-NMPC algorithm so that the lengths of the finite time horizons are chosen in order to ensure the asymptotic stability of the computed feedback controls. The proposed method is successfully tested by numerical examples.']
#bestClass = computeClassLikelihood(input,classes,p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))


#train_out = load_data("train_out.csv")
#final_test = load_data("test_in.csv")
trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset2, train_out2,2)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
# input = ['170', 'In this article a stabilizing feedback control is computed for a semilinear parabolic partial differential equation utilizing a nonlinear model predictive (NMPC) method. In each level of the NMPC algorithm the finite time horizon open loop problem is solved by a reduced-order strategy based on proper orthogonal decomposition (POD). A stability analysis is derived for the combined POD-NMPC algorithm so that the lengths of the finite time horizons are chosen in order to ensure the asymptotic stability of the computed feedback controls. The proposed method is successfully tested by numerical examples.']
#bestClass = computeClassLikelihood(input,classes,p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))


#train_out = load_data("train_out.csv")
#final_test = load_data("test_in.csv")
trainingSet, train_out, testSet, test_out = nb.separateDataForCrossValidation(dataset3, train_out3,3)
classes = nb.get_classes(trainingSet, train_out, common_words)
p_ys = nb.get_py(train_out, classes)
print(p_ys)
# input = ['170', 'In this article a stabilizing feedback control is computed for a semilinear parabolic partial differential equation utilizing a nonlinear model predictive (NMPC) method. In each level of the NMPC algorithm the finite time horizon open loop problem is solved by a reduced-order strategy based on proper orthogonal decomposition (POD). A stability analysis is derived for the combined POD-NMPC algorithm so that the lengths of the finite time horizons are chosen in order to ensure the asymptotic stability of the computed feedback controls. The proposed method is successfully tested by numerical examples.']
#bestClass = computeClassLikelihood(input,classes,p_ys)
predictions = nb.getPredictions(testSet,classes,p_ys,common_words)
val.append(nb.evaluate(predictions,test_out))
predictions = nb.getPredictions(trainingSet, classes, p_ys,common_words)
train.append(nb.evaluate(predictions, train_out))

print(val)
print(train)
print (statistics.mean(val))
print (statistics.mean(train))
print (statistics.stdev(val))
print (statistics.stdev(train))