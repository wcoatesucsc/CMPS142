# random_forest.py
# A random forest algorithm for machine learning

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

# Function to print out various metrics of the SVM models
def output_metrics(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Precision:")
    print(precision_score(y_true, y_pred, average=None))
    print("Micro-average Precision:")
    print(precision_score(y_true, y_pred, average='micro'))
    print("Weighted-average Precision")
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

def main():
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42, verbose = 2)

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

    with open("vectors_scores_20_test.txt") as test:
        test_data = test.read().splitlines()

    testX = []
    testY = []
    for line in test_data:
        vector_list = line.split()
        vector_list = [float(component) for component in vector_list]
        testX.append(vector_list[:len(vector_list) - 1])
        testY.append(int(vector_list[len(vector_list)-1]))

    print("fitting random forest...")
    rf.fit(realX, realY)
    print("random forest fit")

    print("predicting on test set...")
    predictions = rf.predict(testX)
    print("predictions made")

    print("printing metrics")
    output_metrics(testY, predictions)
    print("done checking metrics")

    train.close()
    train2.close()
    test.close()

main()