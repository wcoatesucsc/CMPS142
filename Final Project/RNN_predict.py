from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding


from keras.layers import LSTM
from keras.utils import np_utils
from keras.models import Model


from keras.models import model_from_json
import numpy
import os

import nltk
import csv
import string
import sys

from joblib import dump, load

model = load("RNN_model.joblib")

testcorpus = []

with open(sys.argv[1], 'r', newline='') as testincsvfile:
#with open('test.csv', 'r', newline='') as incsvfile:
  csv_reader = csv.reader(testincsvfile, delimiter=",")
  rownum = 0 
  for row in csv_reader:
    rownum += 1
    if(rownum != 1): 
      phrase = row[2]
      testcorpus.append(phrase)
        #print(phrase)
        #labels.append(int(row[3]))
  print(len(testcorpus))
    
    
    
newReviews=['i hate this movie','horrible movie, worst director ever sksksksdjjjj','funny chick flick', ' i love this movie', 'meh', 'hi']

#Prep the list of movie review inputs for the model

#docs=newReviews
docs=testcorpus
###################

#Prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
#Integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

#Pad documents to a max length of 7 words
max_length = 7
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

###################
#preparedNewReviews=padded_docs
preparedTestCorpus=padded_docs

#Make the predictions
#ylabels = model.predict_classes(preparedNewReviews) #ylabels is a list of numbers
ylabels = model.predict_classes(preparedTestCorpus) #ylabels is a list of numbers

#Show the raw text input reviews and the predicted sentiment class labels.
#for i in range(len(newReviews)):
#   print("X=%s, Predicted=%s" % (newReviews[i], ylabels[i]))

test_out = open("RNN_Test_CoatesWilliamsWynd_predictions.csv", "w", newline='')
csv_writer = csv.writer(test_out, delimiter=",")
csv_writer.writerow(["PhraseId","Sentiment"])

test_csv = open(sys.argv[1], "r", newline='')
#test_csv = open("test.csv", "r", newline='')
#testincsvfile.seek(0)
csv_reader = csv.reader(test_csv, delimiter=",")

rownum = 0
for row in csv_reader:
  #if(rownum != 0 and rownum <= len(ylabels)):
  if(rownum != 0):
    csv_writer.writerow([row[0], ylabels[rownum - 1]])
  rownum += 1