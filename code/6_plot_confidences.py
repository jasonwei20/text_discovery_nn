# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# getting the differences for each chapter

from nlp_utils import *

if __name__ == "__main__":

	i_lines = open("processed_data/nn/i_all.txt", 'r').readlines()
	p_lines = open("processed_data/nn/p_all.txt", 'r').readlines()

	chap_to_side_averages = {}
	for chapter in range(0, 9):
		chap_to_side_averages[chapter] = {'i':{'y':[]}, 'p':{'y':[]}}

	prediction_lines = open("outputs/predictions_logistic.csv", 'r').readlines()[1:]
	line_to_prediction = {}
	for pred_line in prediction_lines:
		parts = pred_line.split(',')
		pred_prob = float(parts[0])
		line = parts[-1]
		line_to_prediction[line] = pred_prob
	print(len(line_to_prediction))

	for i_line in i_lines:
		parts = i_line.split('\t')
		chap = int(i_line[0])
		line = parts[1]
		if line in line_to_prediction:
			chap_to_side_averages[chap]['i']['y'].append(line_to_prediction[line])

	for p_line in p_lines:
		parts = p_line.split('\t')
		chap = int(p_line[0])
		line = parts[1]
		if line in line_to_prediction:
			chap_to_side_averages[chap]['p']['y'].append(line_to_prediction[line])

	for chap in chap_to_side_averages:

		chap_to_side_averages[chap]['i']['x'] = range(0, len(chap_to_side_averages[chap]['i']['y']))
		chap_to_side_averages[chap]['i']['x'] = [float(value)/len(chap_to_side_averages[chap]['i']['x']) for value in chap_to_side_averages[chap]['i']['x']]
		chap_to_side_averages[chap]['p']['x'] = range(0, len(chap_to_side_averages[chap]['p']['y']))
		chap_to_side_averages[chap]['p']['x'] = [float(value)/len(chap_to_side_averages[chap]['p']['x']) for value in chap_to_side_averages[chap]['p']['x']]

		fig, ax = plt.subplots()
		output_path = 'outputs/stories/'+str(chap)+"_story.png"

		x_i = chap_to_side_averages[chap]['i']['x']
		y_i = chap_to_side_averages[chap]['i']['y']
		x_p = chap_to_side_averages[chap]['p']['x']
		y_p = chap_to_side_averages[chap]['p']['y']

		ax.scatter(x_i, y_i, color='b', marker='o', s=0.7, label='Israeli Narrative')
		ax.scatter(x_p, y_p, color='r', marker='o', s=0.7, label='Palestinian Narrative')
		ax.set_title("Narrative Disparity in Chapter " + str(chap+1), fontsize=6.5)

		for i in range(len(x_i)):
			ax.annotate(str(i), (x_i[i], y_i[i]), fontsize = 1)

		for i in range(len(x_p)):
			ax.annotate(str(i), (x_p[i], y_p[i]), fontsize = 1)


		plt.legend(prop={'size': 4})
		plt.savefig(output_path, dpi=1000)
		print('image saved in', output_path)
		plt.clf()














