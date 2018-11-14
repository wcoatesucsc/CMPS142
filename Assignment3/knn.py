# K Nearest Neighbor Implementation for CMPS 142 Assignment 3
# Author: Will Coates, Jacob Wynd


import csv # used to read in test/training data
import math
import sys, getopt # used to read command line options


# KNN Algorithm:
# -Store the distance from x to each point in pointsList
# -Sort pointsList in ascending order by distance from x
# -Start with predicted label = 0 (that's how x is passed in)
# -add the labels of the first k points in the sorted pointsList
#  to the predicted label
# -return the sign of the predicted label
def knn_predict(pointsList, k, x, method):
    # implement KNN algorithm
    dist = []
    label = 0

    reportedDistance = False

    for point in pointsList:
        if method == 'L2':
            d = distance_L2(x, point)

        elif method == 'L1':
            d = distance_L1(point, x)
        else:
            d = distance_Linf(point, x)

        dist.append([d, point.get('label')])
        
    distSorted = sorted(dist, key = lambda q: q[0])
    for i in range(0, k):
        label += int(distSorted[i][1])
        
    if(label > 0):
        label = 1
    elif(label < 0):
        label = -1
    if(label == 0):
        print('Please make K an odd number')
        sys.exit(2)
    return label




# Measures L2 distance (Euclidean distance) between two points
# Formula: sqrt((difference of each coordinate)^2)
def distance_L2(point1, point2):
	# iterate over both points, summing the differences
	# of their coordinates squared
        sum = 0
        for key, value in point1.items():
		if key == 'label':
			continue
                sum += pow((int(point1[key]) - int(point2[key])), 2)
	
	return math.sqrt(sum) 

# Measures L1 distance ( taxicab distance ) between two points
# Formula: sum(abs(difference of each coordinate))
def distance_L1(point1, point2):
	sum = 0
        for key, value in point1.items():
		if key == 'label':
			continue
                sum += abs((int(point1[key]) - int(point2[key])))
        return sum

# Measures Linf distance ( chessboard distance ) between two points
# Formula: max(abs(pi - qi))
def distance_Linf(point1, point2):
	max = 0
        
        for key, value in point1.items():
		if key == 'label':
			continue
                dist = abs((int(point1[key]) - int(point2[key])))
                if dist > max:
			max = dist
	return max


#
# Main Function
#

# ========================================================
# Get options for K and method
# ========================================================
k = 0
method = ""
	
try:
	opts, args = getopt.getopt(sys.argv[1:], 'K:m:', ['K=', 'method='])

except getopt.GetoptError:
	print ('Usage: knn.py --K n --method [L1, L2, or Linf]')
	sys.exit(1)

for opt, arg in opts:
	if opt in ('--K'):
		k = int(arg)
	elif opt in ('--method'):
                if(arg == "L1" or arg == "L2" or arg == "Linf"):
			method = arg
		else:
			print ('Usage: knn.py --K n --method [L1, L2, or Linf]')
			sys.exit(1)

# =========================================================
# Print out some information about this run
# =========================================================
print("K = " + str(k))
print("Distance method = " + method)
# =========================================================
# Read in and store the data as a list of points, each of which
# is a dictionary containing its coordinates and label
# =========================================================

pointsList = []

with open('knn_train.csv', 'rb') as traindata:
	reader = csv.DictReader(traindata)
	for row in reader:
		# create a dictionary for this point
		point = {'f1': row['f1'], 'f2': row[' f2'], 'f3': row[' f3'], 'f4': row[' f4'], 'label': row[' label']}
                pointsList.append(point)

traindata.close()


# =========================================================
# Use the trained data to classify all instances in the test
# set and print their labels
# =========================================================
with open('knn_test.csv', 'rb') as testdata:
	reader = csv.DictReader(testdata)
	index = 1 
	for row in reader:
		# create a dictionary for this point
		point = {'f1': row['f1'], 'f2': row[' f2'], 'f3': row[' f3'], 'f4': row[' f4'], 'label': row[' label']}
                print("Test Instance " + str(index) + " True Label = " + str(point.get("label")) + " Predicted Label = " + str(knn_predict(pointsList, k, point, method)))
		index += 1

testdata.close()





