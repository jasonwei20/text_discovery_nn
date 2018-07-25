# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# Some visual analysis for analyzing historical bias

from nlp_utils import *

def find_notable_lines(line_numbers):

	notable_line_numbers = []
	#notable if 3/7 or 2/4
	for line_number in line_numbers:
		for i in range(line_number-2, line_number+3):
			if not i == line_number and i in line_numbers:
				notable_line_numbers.append(line_number)
				break
	return notable_line_numbers

def get_clusters(line_numbers):

	idx = 0
	sets = []

	while idx < len(line_numbers)-2:
		set_ = [line_numbers[idx]]
		while line_numbers[idx+1] <= line_numbers[idx] + 2:
			set_.append(line_numbers[idx+1])
			idx += 1
		sets.append(set_)
		idx += 1
	return sets

def get_word_freq(word, line_numbers, all_lines):
	occurences = 0
	for line_number in line_numbers:
		line = all_lines[line_number]
		occurences += line.count(word)
	return occurences/len(line_numbers)

if __name__ == "__main__":

	all_lines_path = "processed_data/all_lines.txt"
	group_to_data_path = 'outputs/group_to_data.p'
	output_path = "outputs/analysis_freq.jpg"

	all_lines = open(all_lines_path, 'r').readlines()
	group_to_data = load_obj(group_to_data_path)
	group_to_line_numbers = {}
	group_to_notable_lines = {}
	groups = list(group_to_data.keys())

	#plotting the notable lines
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
	fig, ax = plt.subplots()

	#getting the lines numbers
	for i in range(len(groups)):
		group = groups[i]
		line_numbers = []
		for prediction in group_to_data[group]:
			line = prediction[3]
			line_number = find_line_number(line, all_lines)
			if line_number is not None:
				line_numbers.append(line_number)
		line_numbers = sorted(line_numbers)
		group_to_line_numbers[group] = line_numbers

	#getting the notable lines and clusters
	for group in groups:
		line_numbers = group_to_line_numbers[group]
		notable_lines = find_notable_lines(line_numbers)
		print(group, len(notable_lines), "in clusters out of", len(line_numbers))
		group_to_notable_lines[group] = notable_lines

		writer_path = "outputs/clusters_" + group + ".txt"
		writer = open(writer_path, 'w')
		clusters = get_clusters(notable_lines)
		for cluster in clusters:
			for line_number in cluster:
				writer.write(str(line_number) + "\t" + all_lines[line_number])
			writer.write("\n")
		writer.close()

		print("freq of british", get_word_freq('british', notable_lines, all_lines))

	#plotting it
	print("plotting the notable lines...")
	for i in range(len(groups)):

		group = groups[i]
		color = colors[i]
		line_numbers = group_to_notable_lines[group]

		labels = line_numbers
		x = line_numbers
		y = np.zeros_like(line_numbers) + float(i)
		size = 0.01
		ax.scatter(x, y, color=color, marker='o', s=size, label=group)
		ax.set_title("Selection of Sentences from Israeli and Palestinian Narratives", fontsize=6.5)
		print(group, color)
		for j, txt in enumerate(labels):
			ax.annotate(txt, (x[j], y[j]), fontsize = 0.02)
		#print(group, line_numbers)
	plt.savefig(output_path, dpi=1500)

	












