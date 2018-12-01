#
# Messing around with the scikit-learn library to implement
# SVM multi-class classifiers
# Author: Will Coates
# Started by copy-pasting from:
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py
#


#
# To easily apply this analysis to our data, we need to generate
# a "dictionary-like object" (numpy array??) with the following
# attributes:
# data:   an array of the feature vectors (arrays themselves)
# target: an array of the classes the feature vectors correspond to 
#


print(__doc__)

#quiets a deprecation warning from "cloudpickle", which I can't find on my machine
import imp 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# for some reason the target is an array
import array

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()

# read our training data into a list
training_data = []
with open("vectors_scores.txt") as vectors_scores:
	training_data = vectors_scores.read().splitlines()

realX = []
realY = []
for line in training_data:
	realX.append(line[:len(line) - 1])
	realY.append(int(line[len(line) - 1]))

print(realY)

print(iris)

# Take the first two features. We could avoid this by using a two-dim dataset
#X = iris.data[:, :2]
X = iris.data[:, :3]
y = iris.target

print(y)

# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)



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
for clf, title, in zip(models, titles):
    #plot_contours(ax, clf, xx, yy,
    #              cmap=plt.cm.coolwarm, alpha=0.8)
    #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    #ax.set_xlim(xx.min(), xx.max())
    #ax.set_ylim(yy.min(), yy.max())
    #ax.set_xlabel('Sepal length')
    #ax.set_ylabel('Sepal width')
    #ax.set_xticks(())
    #ax.set_yticks(())
    #ax.set_title(title)

    print("Model: " + title + " Prediction: [1.0, 2.0, 3.0]: " + str(clf.predict([[1.0, 2.0, 3.0]])))

#plt.show()
