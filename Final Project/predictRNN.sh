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

echo "Predict with RNN model"
python3.6 RNN_predict.py $PREDICTFILE
echo "Done predicting"

echo "Predictions in CoatesWilliamsWynd_predictions.csv"

echo "All done! :)"

