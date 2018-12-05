import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from joblib import dump, load
import pandas as pd
import csv
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

print("loading trainedForest.joblib...")
rf = load('trainedForest.joblib')
print("trained forest loaded")

with open("test_vectors.txt") as test:
    test_data = test.read().splitlines()

testX = []
for line in test_data:
    vector_list = line.split()
    vector_list = [float(component) for component in vector_list]
    testX.append(vector_list)
#    testY.append(int(vector_list[len(vector_list)-1]))

print("predicting on trained forest")
predictions = rf.predict(testX)
print("predictions complete")

# read test.csv
# take first value of all lines except for the first one
# create a new file whose output is the phrase idea taken from first col of test.csv
# followed by the prediction

#test_csv = pd.read_csv("testset_1.csv", sep=',', header=None)
#phrase_id = test_csv.PhraseId

test_csv = open("testset_1.csv", "r", newline='')


test_out = open("CoatesWilliamsWynd_predictions.csv", "w", newline='')
csv_writer = csv.writer(test_out, delimiter=",")
csv_writer.writerow(["PhraseId","Sentiment"])

csv_reader = csv.reader(test_csv, delimiter=",")

rownum = 0
for row in csv_reader:
	if(rownum != 0 and rownum <= len(predictions)):
		csv_writer.writerow([row[0], predictions[rownum - 1]])
	rownum += 1

#for i in range(len(predictions)):
#    csv_writer.writerow([[i], predictions[i]])



    #must avoid off by one acception

#print("printing metrics")
#output_metrics(testY, predictions)
#print("done checking metrics")

test.close()
