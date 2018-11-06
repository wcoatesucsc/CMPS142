# K Nearest Neighbor Implementation for CMPS 142 Assignment 3
# Author: Will Coates (so far)


import csv # used to read in test/training data
import math

# Measures L2 distance (Euclidean distance) between two points
# Formula: sqrt((difference of each coordinate)^2)
def distanceL2(point1, point2):
#        print("Point 1:")
#        print(point1)
#        print("Point 2:")
#        print(point2)

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
def distanceL1(point1, point2):
	sum = 0
        for key, value in point1.items():
		if key == 'label':
			continue
                sum += abs((int(point1[key]) - int(point2[key])))
        return sum

# Measures Linf distance ( chessboard distance ) between two points
# Formula: max(abs(pi - qi))
def distanceLinf(point1, point2):
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

print("hello, python!")

# make a list of points, each of which is a dictionary?
pointsList = []

# Read in and store the data
with open('knn_test.csv', 'rb') as testdata:
	reader = csv.DictReader(testdata)
	for row in reader:
		print( row['f1'], row[' f2'], row[' f3'], row[' f4'], row[' label']) 
		# create a dictionary for this point
		point = {'f1': row['f1'], 'f2': row[' f2'], 'f3': row[' f3'], 'f4': row[' f3'], 'label': row[' label']}
                pointsList.append(point)


print("Point 1:")
print(pointsList[0])
print("Point 2:")
print(pointsList[1])
print("L2 distance = " + str(distanceL2(pointsList[0], pointsList[1])))
print("L1 distance = " + str(distanceL1(pointsList[0], pointsList[1])))
print("Linf distance = " + str(distanceLinf(pointsList[0], pointsList[1])))







