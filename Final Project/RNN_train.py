#Nicholas Williams
#ID: 1597706
#UCSC CMPS 142 Fall 2018
#Sentiment Analysis on Movie Reviews
#12/5/2018

#This code trains an LSTM using transfer learning on the
#word embeddings using the pre-trained GloVe twitter word embeddings.

#THIS CODE IS UNFINISHED BECAUSE IT NEEDS TO TRAIN ON THE FULL SIZED
#TRAINING SET.

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

from joblib import dump, load

#Lists to hold the text review strings, and the integer sentiment labels y.
corpus=[] #Python list
labels=([]) #numpy array

#Open csv, read phrases into a string array, and output labels into an array.
with open('train_undersampled.csv', 'r', newline='') as incsvfile:
    csv_reader = csv.reader(incsvfile, delimiter=",")
    rownum = 0
    for row in csv_reader:
        rownum += 1
        if(rownum != 1):
          phrase = row[2]
          corpus.append(phrase)
          labels.append(int(row[3]))

#These are just the garbage csv headers
corpus.pop(0)
labels.pop(0)

#Define documents
docs = corpus

#Define dataset size before split
docs=docs[0:4000]
labels=labels[0:4000]
labels = np_utils.to_categorical(labels) #needs to be one hot encoded


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

#Load the whole embedding into memory
embeddings_index = dict()
f = open('glove.twitter.27B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

#Create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    
#Assign training, validation, and test data.
X_train=padded_docs[0:4000]
y_train=labels[0:4000]

X_valid=padded_docs[4000:5000]
y_valid=labels[4000:5000]

X_test=padded_docs[5000:6000]
y_test=labels[5000:6000]

    
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=7, trainable=True)
model.add(e)
#LSTM layer with 100 dimension, and the return_sequences insures full
#connection to the next layer instead of only the last sequence(time) step.
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(5, activation='softmax'))
print(model.summary())


#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#Print a summary of the model.
print(model.summary())

batch_size = 4
num_epochs = 3

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs) #Getting a dimension error here

#Print the performance metrics
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
#print('Test fmeasure:', scores[2])
#print('Test precision:', scores[3])
#print('Test recall:', scores[4])

print("Saving model to the disk...")
dump(model, 'RNN_model.joblib')
print("Model created")
