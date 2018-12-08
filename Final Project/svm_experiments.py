# Implementing SVM multi-class classifiers
# with the scikit-learn library

# Author: Will Coates
# Started by copy-pasting from:
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py

print(__doc__)

import sys

#quiets a deprecation warning from "cloudpickle", which I can't find on my machine
import imp 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import *
from joblib import dump, load
# for some reason the target is an array
import array

# set each to True if we want to train/predict this model
lksvc = False
lsvc = True
rbfsvc = False
polysvc = False

# Function to print out various metrics of the SVM models
def output_metrics(y_true, y_pred):
	print("Confusion Matrix:")
	print(confusion_matrix(y_true, y_pred))
	print("Accuracy:")
	print(accuracy_score(y_true, y_pred))
	print("Precision:")
	print(precision_score(y_true, y_pred, average=None))
	print("Micro-average Precision:")
	print(precision_score(y_true, y_pred, average='micro'))
	print("Weighted-average Precision:")
	print(precision_score(y_true, y_pred, average='weighted'))
	print("Recall:")
	print(recall_score(y_true, y_pred, average=None))
	print("Micro-average Recall:")
	print(recall_score(y_true, y_pred, average='micro'))
	print("Weighted-average Recall:")
	print(recall_score(y_true, y_pred, average='weighted'))
	print("F1 Score:")
	print(f1_score(y_true, y_pred, average=None))
	print("Micro-average F1 Score:")
	print(f1_score(y_true, y_pred, average='micro'))
	print("Weighted-average F1 Score:")
	print(f1_score(y_true, y_pred, average='weighted'))

# read our training data into a list
training_data = []
with open("vectors_scores_60.txt") as train:
#with open(sys.argv[1]) as train:
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

# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors
print("Initializing models")

C = 1 # SVM regularization parameter
max_iters = 1000 
#class_weights = {0: 1, 1:4, 2:10, 3:4, 4:1}
#class_weights = {0:10, 1:4, 2:1, 3:4, 4:10}
# try 'balanced' for the class_weights instead
class_weights = 'balanced'
#class_weights = None

if(lksvc):
	# experimenting with class_weight to handle imbalanced
	# data
	linear_kernel_svc = svm.SVC(kernel='linear',
				class_weight=class_weights,
				max_iter=max_iters,
				verbose=True,
				C=C)
if(lsvc):
	linear_svc        = svm.LinearSVC(class_weight=class_weights,
				verbose=True,
				C=C)
if(rbfsvc):
	rbf_kernel_svc    = svm.SVC(kernel='rbf', 
				    class_weight=class_weights,
				    gamma=0.7, 
				    max_iter=max_iters,
				    verbose=True,
				    C=C)
if(polysvc):
	poly_kernel_svc   = svm.SVC(kernel='poly', 
				    class_weight=class_weights,
				    max_iter=max_iters,
				    degree=3, 
				    verbose=True,
				    C=C)

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

print("DUMPING =========================") 

dump(lsvc, "linearkernelsvc_model.joblib")

loaded_model = load("linearkernelsvc_model.joblib")

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

if(lksvc):
	lksvc_cross_output = []
if(lsvc):
	lsvc_cross_output = []
if(rbfsvc):
	rbfsvc_cross_output = []
if(polysvc):
	polysvc_cross_output = []

print("PREDICTING========================")
if(lksvc):
	lksvc_cross_output = (linear_kernel_svc.predict(crossX))
	lksvc_predictions = open("lksvc_predictions.txt", "w")
	for line in lksvc_cross_output:
		lksvc_predictions.write(str(line) +  "\n")
	lksvc_predictions.close()
if(lsvc):
	lsvc_cross_output = (linear_svc.predict(crossX))
	lsvc_predictions = open("lsvc_predictions.txt", "w")
	for line in lsvc_cross_output:
		lsvc_predictions.write(str(line) +  "\n")
	lsvc_predictions.close()
if(rbfsvc):
	rbfsvc_cross_output = (rbf_kernel_svc.predict(crossX)) 
	rbfsvc_predictions = open("rbfsvc_predictions.txt", "w")
	for line in rbfsvc_cross_output:
		rbfsvc_predictions.write(str(line) +  "\n")
	rbfsvc_predictions.close()
if(polysvc):
	polysvc_cross_output = (poly_kernel_svc.predict(crossX)) 
	polysvc_predictions = open("polysvc_predictions.txt", "w")
	for line in polysvc_cross_output:
		polysvc_predictions.write(str(line) + "\n")
	polysvc_predictions.close()

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
