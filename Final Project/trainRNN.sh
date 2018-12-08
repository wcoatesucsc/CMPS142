#!/bin/bash


# Modules that need to be installed (not comprehensive yet)
# pandas, joblib, numpy, sklearn, ....
#python -m pip install --upgrade pip
#pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose scikit-learn


# take in the file we're predicting on as an option

echo "Training RNN"

# clean the test data, output each cleaned phrase on its own line
# in unlabeled_cleaned_test.txt

echo "unzipping"
bunzip2 glove.twitter.27B.100d.txt.bz2

echo "Training RNN model"
python3.6 RNN_train.py
echo "Done training"

echo "rezipping"
bzip2 glove.twitter.27B.100d.txt

echo "run predictRNN.sh when ready"

echo "All done! :)"

