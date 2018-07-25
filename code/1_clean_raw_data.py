#Jason Wei
#Cleaning raw text files
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()
len_threshold = 50

def get_only_chars(line):
	line = line.lower()
	only_chars_line = []

	for char in line:
		if char in 'qwertyuiopasdfghjklzxcvbnm ':
			only_chars_line.append(char)

	only_chars_line = ''.join(only_chars_line)


	if len(only_chars_line) > 20:

		while only_chars_line[0] == ' ':
			only_chars_line = only_chars_line[1:]

		while only_chars_line[-1] == ' ':
			only_chars_line = only_chars_line[:-1]

		return only_chars_line
	else:
		return ''

def delete_spaces(line):
	return re.sub(' +',' ',line)

def long_enough(line, threshold):
	if len(line) > threshold:
		return True
	return False

if __name__ == "__main__":
	lines = open(args.input_path, 'r',  encoding='latin-1').readlines()

	output_lines = []
	for line in lines:
		output_line = get_only_chars(line)
		output_line = delete_spaces(output_line)
	
		if long_enough(output_line, len_threshold):
			output_lines.append(output_line)

	writer = open(args.output_path, 'w')
	for line in output_lines:
		writer.write(line+'\n')











