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


# Machine Learning Libraries Imports
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

import csv
import string


clean_corpus = []


def clean_phrase(phrase):
	# convert phrase to lowercase
	phraseLower = phrase.lower()
	# tokenize the phrase (now it's a list of strings)
	tokens = phraseLower.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]

	# remove other non-alphabetic tokens
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [word for word in tokens if not word in stop_words]
        
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	# 
	# OUR ADDITIONS TO TUTORIAL CLEANING METHODS
	# 
	# lemmatization:
	tokens = [lemmatizer.lemmatize(word) for word in tokens]

	for word in range(len(tokens)):
		clean_corpus.append(tokens[word])
			
	# converting list of tokens back to a string
	separator = " "
	cleanedPhrase = separator.join(tokens)
	return cleanedPhrase
			


def main():
	print("Cleaning data! :)")
	# simultaneously read from train.csv and write to clean_train.csv
	with open('train.csv', 'r', newline='') as incsvfile:
		with open('clean_train.csv', 'w', newline='') as outcsvfile:
			csv_reader = csv.reader(incsvfile, delimiter=",")
			
			csv_writer = csv.writer(outcsvfile, delimiter=",")
			# write header row
			# NOTE! WHEN RUNNING ON TEST DATA EXCLUDE SENTIMENT!
			#csv_writer.writerow(['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']) 

			rownum = 0	
			for row in csv_reader:
				rownum += 1
				print("row: " + str(rownum))
				# grab phrase and apply cleaning procedures to it
				phrase = row[2]
				cleanedPhrase = clean_phrase(phrase)
				# write the same row (except with a clean phrase) to the output file
				#NOTE! WHEN RUNNING ON TEST DATA EXCLUDE ROW[3]!!
				csv_writer.writerow([row[0], row[1], cleanedPhrase, row[3]])

	incsvfile.close()
	outcsvfile.close()
	
	#print(clean_corpus)
	fdist = FreqDist(clean_corpus)
	print("Words that appear only once:")
	print(fdist.hapaxes())

        # should write to csv here, when we can filter out hapaxes,
	# instead of above

main()
