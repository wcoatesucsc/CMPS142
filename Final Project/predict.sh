#!/bin/bash



# Modules that need to be installed (not comprehensive yet)
# pandas, joblib, numpy, sklearn, ....



# Script to take in CSV file in test.csv format,
# perform data pre-processing, and predict using whichever
# model we prefer
echo "Predicting on test.csv!"

# clean the test data, output each cleaned phrase on its own line
# in unlabeled_cleaned_test.txt
echo "Cleaning test data:"
python3.6 data_clean.py -m test
echo "Done cleaning test data"

echo "Generating a vector for each phrase with Fasttext"
FastTextDir/fasttext print-sentence-vectors model.bin < unlabeled_cleaned_test.txt > test_vectors.txt
echo "Done generating sentence vectors"

# Unzips Jacob Wynd's trained Random Forest model for prediction
echo "Unzipping trained Random Forest Model"
bunzip2 trainedForest.joblib.bz2
echo "Done unzipping"

echo "Predict with Random Forest Model"
python3.6 random_forest_predict.py
echo "Done predicting"

echo "Clean up intermediate files"
rm unlabeled_cleaned_test.txt test_vectors.txt

echo "Rezip random forest classifier"
bzip2 trainedForest.joblib

echo "All done! :)"

