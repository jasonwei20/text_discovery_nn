# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# getting the differences for each chapter

from nlp_utils import *

if __name__ == "__main__":

	#all the data sources
	vec_length = 300
	tsne_output_path = 'outputs/chapter_tsne.png'
	i_lines = open("processed_data/nn/i_all.txt", 'r').readlines()
	p_lines = open("processed_data/nn/p_all.txt", 'r').readlines()
	print(len(i_lines)+len(p_lines))

	chap_to_side_averages = {}
	for chapter in range(0, 9):
		chap_to_side_averages[chapter] = {'i':[], 'p':[]}

	prediction_lines = open("outputs/predictions_logistic.csv", 'r').readlines()[1:]
	line_to_prediction = {}
	for pred_line in prediction_lines:
		parts = pred_line.split(',')
		print(parts[0])
		pred_prob = float(parts[0])
		line = parts[-1]
		line_to_prediction[line] = pred_prob
	print(len(line_to_prediction))

	for i_line in i_lines:
		parts = i_line.split('\t')
		chap = int(i_line[0])
		line = parts[1]
		if line in line_to_prediction:
			chap_to_side_averages[chap]['i'].append(line_to_prediction[line])

	for p_line in p_lines:
		parts = p_line.split('\t')
		chap = int(p_line[0])
		line = parts[1]
		if line in line_to_prediction:
			chap_to_side_averages[chap]['p'].append(line_to_prediction[line])

	for key in chap_to_side_averages:
		i_list = chap_to_side_averages[key]['i']
		p_list = chap_to_side_averages[key]['p']
		#print(1-sum(i_list)/len(i_list))
		print(sum(p_list)/len(p_list))
























