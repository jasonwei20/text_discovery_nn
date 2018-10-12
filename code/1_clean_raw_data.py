#Jason Wei
#Cleaning raw text files
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

def get_only_chars(line):

	clean_line = ""

	line = line.lower()
	line = line.replace("-", " ") #replace hyphens with spaces
	line = line.replace("\t", " ")
	line = line.replace("\n", " ")

	for char in line:
		if char in 'qwertyuiopasdfghjklzxcvbnm ':
			clean_line += char
		else:
			clean_line += ' '

	return clean_line

def delete_spaces(line):
	return re.sub(' +',' ',line)


if __name__ == "__main__":

	line = open(args.input_path, 'r',  encoding='latin-1').read()
	line = get_only_chars(line) #clean up the chars
	line = delete_spaces(line)

	writer = open(args.output_path, "w")
	writer.write(line)
	print(len(line.split(" ")))









