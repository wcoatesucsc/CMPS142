#
# Cleaning and Preprocessing the Movie Review Data:
# Authors: Will Coates, Nick Williams, Jacob Wynd
# For cleaning, following part of a tutorial by Jason Brownlee:
# https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/
#


#
# Input:  .csv file with PhraseID, SentenceID, Phrase[, label]
# Output: .csv file with PhraseID, SentenceID, Cleaned Phrase[, label]

# Read input file one line at a time, apply cleaning procedures to the
# Phrase column, and write the entire row (with a cleaned Phrase) to the 
# output file, one line at a time.
#
import imp
# Prints error if trying to run with no options
import sys, getopt
usage = "data_clean.py -m <test or train> -f inputfile.csv"
if len(sys.argv) <= 2:
	print(usage)
	sys.exit(2)

# Machine Learning Libraries Imports
import operator
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

import csv
import string

clean_corpus_words = []
clean_corpus_phrases = []

# takes in a string (phrase), tokenizes it, performs
# various data preprocessing, rejoins it as a string,
# and returns the cleaned string
def clean_phrase(phrase):
	# convert phrase to lowercase
	phraseLower = phrase.lower()
	# tokenize the phrase (now it's a list of strings)
	tokens = phraseLower.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]

	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]

	# remove other non-alphabetic tokens
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [word for word in tokens if not word in stop_words]
        # # OUR ADDITION TO TUTORIAL CLEANING METHODS
	# lemmatization:
	tokens = [lemmatizer.lemmatize(word) for word in tokens]

	for word in range(len(tokens)):
		clean_corpus_words.append(tokens[word])
			
	# converting list of tokens back to a string
	separator = " "
	cleaned_phrase = separator.join(tokens)
	return cleaned_phrase
		

# takes a phrase and a list of words to filter out of that
# phrase, returns the filtered phrase
def filter_words_from_list(phrase, hapaxes):
	tokens = phrase.split()
	tokens = [word for word in tokens if word not in hapaxes]
	# converting list of tokens back to a string
	separator = " "
	filtered_phrase = separator.join(tokens)
	return filtered_phrase

	
	
def main(argv):
	infilename = ""
	try:
		opts, args = getopt.getopt(argv,"m:f:", ["mode=","file="])
	except getopt.GetoptError:
		print(usage)
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-m", "--mode"):
			if arg == 'train':
				testing = False
			elif arg == 'test':
				testing = True
			else: print(usage)
		elif opt in ("-f", "--file"):
			infilename = arg
			if infilename[-3:] != "csv":
				print(usage)


	print("Cleaning data! :)")
	# also output to a .txt file to play nice with fasttext
	if(testing):
		unlabeled_cleaned_test_txt = open("unlabeled_cleaned_test.txt", "w")
	else:
		labeled_cleaned_train_txt = open("labeled_cleaned_train.txt", "w")
		unlabeled_cleaned_train_txt = open("unlabeled_cleaned_train.txt", "w")

	dictionary_txt = open("dictionary.txt", "w")

	# Read data from train.csv and output a cleaned phrase
	# to each slot in clean_corpus_phrases
	
	if(testing):
		incsvfile = open(infilename, 'r', newline ='')
	else:
		incsvfile = open('train.csv', 'r', newline='') 
		
	csv_reader = csv.reader(incsvfile, delimiter=",")

	rownum = 0	
	for row in csv_reader:
		rownum += 1
		print("Reading row: ", str(rownum), end='\r') 
		# grab phrase and apply cleaning procedures to it
		phrase = row[2]
		cleaned_phrase = clean_phrase(phrase)
		clean_corpus_phrases.append(cleaned_phrase)
	print("Done reading data             ")
		
	# Detect words that only occur once in the entire corpus. We will
	# throw those out
	fdist = FreqDist(clean_corpus_words)
	dictionary_list = set(clean_corpus_words)
	#print(dictionary_list)

	hapaxes = fdist.hapaxes()
	
	# Now that we've found the words that only occur once, filter
	# those out of our cleaned phrases and write those to
	# the output file
	if(testing):
		outcsvfile = open('clean_test.csv', 'w', newline='')
	else:
		outcsvfile = open('clean_train.csv', 'w', newline='')

	csv_writer = csv.writer(outcsvfile, delimiter=",")

	# resetting the input file so we can retain row information
	incsvfile.seek(0)
	rownum = 0
	for row in csv_reader:
		rownum += 1
		print( "Writing row: ", str(rownum), end='\r') 
		# write each cleaned phrase to the output file,
		# filtering out the rare words if they occur
		phrase = clean_corpus_phrases[rownum - 1]
		filtered_phrase = filter_words_from_list(phrase, hapaxes)
		# writing to both output csv and output txt
		if(testing):
			csv_writer.writerow([row[0], filtered_phrase])
		else:
			csv_writer.writerow([row[0], row[1], filtered_phrase, row[3]])

		if(rownum != 1):
			if(testing):
				unlabeled_cleaned_test_txt.write(filtered_phrase+ '\n')
			else:
				score = row[3]
				labeled_cleaned_train_txt.write("__label__" + score + " " + filtered_phrase + '\n')
				unlabeled_cleaned_train_txt.write(filtered_phrase + '\n')
	print("Done writing data              ")	

	# now output dictionary of all words in dataset
	for word in dictionary_list:
		dictionary_txt.write(word + '\n')

	# also output sorted cleaned_train list
	reader = csv.reader(open("clean_train.csv"), delimiter = ",")
	sortedlist = sorted(reader, key=operator.itemgetter(3))
	sorted_clean_train = open("sorted_clean_train.txt", "w")

	for line in sortedlist:
		sorted_clean_train.write(str(line) + "\n")
	sorted_clean_train.close()

	if(not testing):
		print("Cleaned output written to clean_train.csv and labeled_cleaned_train_txt.txt!")
		print("Also wrote output without labels to unlabeled_clean_train.txt, and a dictionary to dictionary.txt")
		print("Have a nice day!")

	incsvfile.close()
	outcsvfile.close()
	if(testing):
		unlabeled_cleaned_test_txt.close()
	else:
		labeled_cleaned_train_txt.close()
		unlabeled_cleaned_train_txt.close()
	dictionary_txt.close()

main(sys.argv[1:])
