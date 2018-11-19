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








def main():
	print("Cleaning data! :)")

	# load the document
	filename = 'train.csv'


	# make all text lowercase
	textLower = text.lower()
	
	# split into tokens by white space
	tokens = textLower.split()

	#
	# Start cleaning and removing tokens
	#

	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]



	print(tokens)


main()
