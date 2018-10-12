# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# Some visual analysis for analyzing historical bias

from nlp_utils import *

def get_plot_vectors(label_dictionary, line_to_vec):

    groups = label_dictionary.keys()

    #create embedding matrix
    lines = []
    for group in groups:
        line_groups = label_dictionary[group]
        lines += line_groups
    line_embeddings = np.zeros((len(lines), 300))
    for i in range(len(lines)):
        line_embeddings[i, :] = line_to_vec[lines[i]]

    #get the tsne for this
    tsne = TSNE(n_components=2).fit_transform(line_embeddings)
    
    return_dict = {}
    counter = 0
    for group in groups:
        x = []
        y = []
        group_size = len(label_dictionary[group])
        for j in range(counter, counter+group_size):
            x.append(tsne[j][0])
            y.append(tsne[j][1])
        return_dict[group] = [label_dictionary[group], x, y]
        counter += group_size
    return return_dict


if __name__ == "__main__":

	#all the data sources
	vec_length = 300
	tsne_output_path = 'chapter_tsne.png'
	line_to_vec, vec_to_line = pickle.load( open('pickles/vec_dicts.p', 'rb') )
	i_lines = open("processed_data/nn/i_all.txt", 'r').readlines()
	p_lines = open("processed_data/nn/p_all.txt", 'r').readlines()
	chapters = ['0', '1', '2', '3']
	group_to_data = {}
	for chapter in chapters:
		for char in ["i_", "p_"]:
			group_to_data[char+chapter] = []

	line_to_line_number = {}

	for i in range(len(i_lines)):
		line = i_lines[i]
		chapter = line.split('\t')[0]
		content = line.split('\t')[1]
		if chapter in chapters:
			group_to_data['i_'+chapter].append(content)
		line_to_line_number[content] = i

	for i in range(len(p_lines)):
		line = p_lines[i]
		chapter = line.split('\t')[0]
		content = line.split('\t')[1]
		if chapter in chapters:
			group_to_data['p_'+chapter].append(content)
		line_to_line_number[content] = i

	#print frequency distribution
	for group in group_to_data:
		print(group, ":", len(group_to_data[group]))

	#get the tsne embeddings
	print("calculating tsne...")
	plot_dict = get_plot_vectors(group_to_data, line_to_vec)
	print("tsne generated")

	groups = list(plot_dict.keys())
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
	markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
	fig, ax = plt.subplots()

	#plot all of them
	for i in range(len(groups)):
		group = groups[i]
		marker = None
		size = None
		if group.split('_')[0] == 'i':
			marker = '.'
			size = 0.15
		elif group.split('_')[0] == 'p':
			marker = 'v'
			size = 0.7
		color = colors[int(group.split('_')[1])]
		labels = plot_dict[group][0]
		x = plot_dict[group][1]
		y = plot_dict[group][2]
		ax.scatter(x, y, color=color, marker=marker, s=size, label=group)
		ax.set_title("Selection of Sentences from Israeli and Palestinian Narratives", fontsize=6.5)
		plt.axis('off')

		#label all the dots
		for i, txt in enumerate(labels):
			ax.annotate(line_to_line_number[txt], (x[i], y[i]), fontsize = 0.5)

	plt.legend(prop={'size': 4})
	plt.savefig(tsne_output_path, dpi=1500)
	print('image saved in', tsne_output_path)
	plt.clf()

	for j in range(len(chapters)):

		fig, ax = plt.subplots()
		for i in range(len(groups)):
			group = groups[i]
			marker = None
			size = None
			if group.split('_')[0] == 'i':
				marker = '.'
				size = 0.15
			elif group.split('_')[0] == 'p':
				marker = 'v'
				size = 0.7
			color = colors[int(group.split('_')[1])]
			if int(group.split('_')[1]) != j:
				color = 'w'
			labels = plot_dict[group][0]
			x = plot_dict[group][1]
			y = plot_dict[group][2]
			ax.scatter(x, y, color=color, marker=marker, s=size, label=group)
			ax.set_title("Selection of Sentences from Israeli and Palestinian Narratives", fontsize=6.5)
			plt.axis('off')

			if int(group.split('_')[1]) == j:
				#label all the dots
				for i, txt in enumerate(labels):
					ax.annotate(line_to_line_number[txt], (x[i], y[i]), fontsize = 0.5)

		plt.legend(prop={'size': 4})
		plt.savefig(str(j) + tsne_output_path, dpi=1500)
		print('image saved in', str(j) + tsne_output_path)
		plt.clf()










