#
# Messing around with the scikit-learn library to implement
# SVM multi-class classifiers
# Author: Will Coates
# Started by copy-pasting from:
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py
#

print(__doc__)

#quiets a deprecation warning from "cloudpickle", which I can't find on my machine
import imp 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import *

# for some reason the target is an array
import array



# set each to True if we want to train/predict this model
lksvc = True 
lsvc = True 
rbfsvc = True 
polysvc = True 

# Function to print out various metrics of the SVM models
def output_metrics(y_true, y_pred):
	print("Confusion Matrix:")
	print(confusion_matrix(y_true, y_pred))
	print("Precision:")
	print(precision_score(y_true, y_pred, average=None))
	print("Recall:")
	print(recall_score(y_true, y_pred, average=None))
	print("F1 Score:")
	print(f1_score(y_true, y_pred, average=None))

# import some data to play with
#iris = datasets.load_iris()

# read our training data into a list
training_data = []
with open("vectors_scores_60.txt") as train:
	training_data = train.read().splitlines()

realX = []
realY = []
for line in training_data:
	vector_list = line.split()
	vector_list = [float(component) for component in vector_list]
	realX.append(vector_list[:len(vector_list) - 1])
	realY.append(int(vector_list[len(vector_list) - 1]))
# also train on the cross-validation set, so an 80-20 split for training-testing
with open("vectors_scores_20_cross.txt") as train2:
	training_data_2 = train2.read().splitlines()
for line in training_data_2:
	vector_list = line.split()
	vector_list = [float(component) for component in vector_list]
	realX.append(vector_list[:len(vector_list) - 1])
	realY.append(int(vector_list[len(vector_list) - 1]))

#print(realY)

#print(iris)

# Take the first two features. We could avoid this by using a two-dim dataset
#X = iris.data[:, :2]

#X = iris.data[:, :3]
#y = iris.target

# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors
print("Initializing models")

C = 1.0  # SVM regularization parameter

if(lksvc):
	linear_kernel_svc = svm.SVC(kernel='linear', C=C)
if(lsvc):
	linear_svc        = svm.LinearSVC(C=C)
if(rbfsvc):
	rbf_kernel_svc    = svm.SVC(kernel='rbf', gamma=0.7, C=C)
if(polysvc):
	poly_kernel_svc   = svm.SVC(kernel='poly', degree=3, C=C)

print("FITTING MODELS ==============================") 
if(lksvc):
	print("Fitting linear_kernel_svc:")
	linear_kernel_svc.fit(realX, realY)
	print("Done fitting linear_kernel_svc:")
if(lsvc):
	print("Fitting linear_svc:")
	linear_svc.fit(realX, realY)
	print("Done fitting linear_svc:")
if(rbfsvc):
	print("Fitting rbf_kernel_svc:")
	rbf_kernel_svc.fit(realX, realY)
	print("Done fitting rbf_kernel_svc:")
if(polysvc):
	print("Fitting poly_kernel_svc:")
	poly_kernel_svc.fit(realX, realY)
	print("Done fitting poly_kernel_svc:")


print("DONE FITTING MODELS =========================") 

# title for the plots

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
#fig, sub = plt.subplots(2, 2)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)

#X0, X1 = X[:, 0], X[:, 1]
#xx, yy = make_meshgrid(X0, X1)

#for clf, title, ax in zip(models, titles, sub.flatten()):

#
# Prediction can go here! We can get a prediction from each of our
# models
#
print("Predicting on cross-validation:")


with open("vectors_scores_20_cross.txt") as cross:
	cross_validation_data = cross.read().splitlines()

crossX = []
crossY = []
for line in cross_validation_data:
	vector_list = line.split()
	vector_list = [float(component) for component in vector_list]
	crossX.append(vector_list[:len(vector_list) - 1])
	crossY.append(int(vector_list[len(vector_list) - 1]))

#with open("cross_output_linear_kernel_svc.txt", "w") as cross_output:


if(lksvc):
	#lksvc_cross_output = open("cross_output_linear_kernel_svc.txt", "w")
	lksvc_cross_output = []
if(lsvc):
	#lsvc_cross_output = open("cross_output_linear_svc.txt", "w")
	lsvc_cross_output = []
if(rbfsvc):
	#rbfsvc_cross_output = open("rbf_kernel_svc.txt", "w")
	rbfsvc_cross_output = []
	
if(polysvc):
	#polysvc_cross_output = open("poly_kernel_svc.txt", "w")
	polysvc_cross_output = []

print("PREDICTING========================")
for instance in crossX:
	#print("crossX line = " + str(instance))
	if(lksvc):
		lksvc_cross_output.append(linear_kernel_svc.predict([instance]))
	if(lsvc):
		lsvc_cross_output.append(linear_svc.predict([instance]))
	if(rbfsvc):
		rbfsvc_cross_output.append(rbf_kernel_svc.predict([instance])) 
	if(polysvc):
		polysvc_cross_output.append(poly_kernel_svc.predict([instance])) 

print("DONE PREDICTING===================")


print("CHECKING METRICS=================")

# output confusion matrix of each prediction against reality
if(lksvc):
	print("LINEAR_KERNEL_SVC===========================")
	output_metrics(crossY, lksvc_cross_output)
	print("============================================\n")
if(lsvc):
	print("LINEAR_SVC==================================")
	output_metrics(crossY, lsvc_cross_output)
	print("============================================\n")
if(rbfsvc):
	print("RBF_KERNEL_SVC==============================")
	output_metrics(crossY, rbfsvc_cross_output)
	print("============================================\n")
if(polysvc):
	print("POLY_KERNEL_SVC=============================")
	output_metrics(crossY, polysvc_cross_output)
	print("============================================\n")

print("DONE CHECKING METRICS============")
#for clf, title, in zip(models, titles):

    #plot_contours(ax, clf, xx, yy,
    #               cmap=plt.cm.coolwarm, alpha=0.8)
    #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #ax.set_xlim(xx.min(), xx.max())
    #ax.set_ylim(yy.min(), yy.max())
    #ax.set_xlabel('Sepal length')
    #ax.set_ylabel('Sepal width')
    #ax.set_xticks(())
    #ax.set_yticks(())
    #ax.set_title(title)    
#    print("Model: " + title + " Prediction: [1.0, 2.0, 3.0]: " + str(clf.predict([[1.0, 2.0, 3.0]])))

#plt.show()

train.close()
cross.close()
