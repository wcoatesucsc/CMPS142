#!/bin/bash


# Modules that need to be installed (not comprehensive yet)
# pandas, joblib, numpy, sklearn, ....
#python -m pip install --upgrade pip
#pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose scikit-learn


# take in the file we're predicting on as an option
USAGE="Usage: $0 predictfile.csv"
PREDICTFILE=""
# Script to take in CSV file in test.csv format,
# perform data pre-processing, and predict using whichever
# model we prefer
if [ "$#" == "0" ]; then
	echo "$USAGE"
	exit 1
fi


if [[ ${1: -3} == "csv" ]]; then
	PREDICTFILE=$1
  else
  	echo "$USAGE"
	exit 1
fi

shift


echo "Predicting on $PREDICTFILE !"

# clean the test data, output each cleaned phrase on its own line
# in unlabeled_cleaned_test.txt
echo "Cleaning test data:"
python3.6 data_clean.py -m test -f $PREDICTFILE
echo "Done cleaning test data"

echo "Generating a vector for each phrase with Fasttext"
FastTextDir/fasttext print-sentence-vectors model.bin < unlabeled_cleaned_test.txt > test_vectors.txt
echo "Done generating sentence vectors"

# Unzips Jacob Wynd's trained Random Forest model for prediction
echo "Unzipping trained Random Forest Model"
bunzip2 trainedForest.joblib.bz2
echo "Done unzipping"

echo "Predict with Random Forest Model"
python3.6 random_forest_predict.py $PREDICTFILE
echo "Done predicting"

echo "Clean up intermediate files"
rm unlabeled_cleaned_test.txt test_vectors.txt

echo "Rezip random forest classifier"
bzip2 trainedForest.joblib

echo "All done! :)"

