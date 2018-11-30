
# Once sentence vectors have been generated for each clean phrase,
# attach their scores to the end of each vector.

# The new file then reads:

# [phrase] vector score

# phrase = usually multiple words, the phrase that the vector corresponds to
#  NOTE: Some phrases are blank, because the original phrase was all stop words
# vector = 100 floating point numbers
# score = integer from 0-4

import csv
import string

def token_is_number(token):
	possible_starts = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]
	return (token[0] in possible_starts)


def main():
	print("I'm gonna append the score to each vector!")
	print("Also will remove the phrase in front and just print the vector to another file")
	# copy each line of the .txt file for later
	with open("sentence_vectors.txt") as sentence_vectors:
		lines = sentence_vectors.read().splitlines()

	with open("sentence_vectors_scores.txt", "w") as sentence_vectors_scores:
		with open("vectors.txt", "w") as vectors:
			with open("clean_train.csv", 'r', newline='') as incsvfile:
				csv_reader = csv.reader(incsvfile, delimiter=",")
				rownum = 0
				for row in csv_reader:
					score = row[3]
					#if(rownum < len(lines) and rownum != 0): 
					if(rownum != 0): 
						# attach scores to the end of lines
						print(lines[rownum - 1] + score, file=sentence_vectors_scores)
						# remove sentences from the beginning of lines and print
						# just the vectors to vectors
						tokens = lines[rownum - 1].split()
						vectors_only = [token for token in tokens if token_is_number(token)]
						separator = " "
						vector_only = separator.join(vectors_only)
						print(vector_only, file=vectors)
					rownum += 1
					




	vectors.close()
	sentence_vectors_scores.close()


main()
