import imp
from joblib import dump, load
import csv
import sys

print("LOADING linearkernelsvc_model.joblib") 
loaded_model = load("linearkernelsvc_model.joblib")
print("MODEL LOADED")


print("READING IN TEST DATA")
with open("test_vectors.txt") as inpredictfile:
	predict_data = inpredictfile.read().splitlines()

crossX = []
for line in predict_data:
	vector_list = line.split()
	vector_list = [float(component) for component in vector_list]
	crossX.append(vector_list[:len(vector_list)])
print("DONE READING IN TEST DATA")

print("PREDICTING for Linear Kernel SVM")
lsvc_cross_output = (loaded_model.predict(crossX))

predictions = []
for line in lsvc_cross_output:
	predictions.append(line)
print("PREDICTION DONE for Linear Kernel SVM")


print("Writing predictions to CoatesWilliamsWynd_predictions_SVM.csv")
#test_csv = open("testset_1.csv", "r", newline='')
test_csv = open(sys.argv[1], "r", newline='')

test_out = open("CoatesWilliamsWynd_predictions_SVM.csv", "w", newline='')
csv_writer = csv.writer(test_out, delimiter=",")
csv_writer.writerow(["PhraseId","Sentiment"])

csv_reader = csv.reader(test_csv, delimiter=",")

rownum = 0
for row in csv_reader:
	if(rownum != 0 and rownum <= len(predictions)):
		csv_writer.writerow([row[0], predictions[rownum - 1]])
	rownum += 1
print("DONE WRITING PREDICTIONS")
