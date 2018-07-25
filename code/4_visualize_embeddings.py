# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# Some visual analysis for analyzing historical bias

from nlp_utils import *

if __name__ == "__main__":

    #all the data sources
    vec_length = 200
    word2vec_path = "word2vec/word2vec_200.p"
    stop_words_path = "processed_data/stop_words.txt"
    predictions_path = "outputs/nn_predictions.csv"
    tsne_output_path = "outputs/tsne.jpg"
    group_to_data_path = 'outputs/group_to_data.p'
    groups = ['mid', 'i_correct', 'i_incorrect', 'p_correct', 'p_incorrect']
    thresholds = [0.2, 0.4, 0.6, 0.8] #we categorize <0.2 as i, between 0.4 and 0.6 as mid, and >0.8 as p
    group_to_output_path = {group : str('outputs/'+group+'.csv') for group in groups}
    group_to_data = {group:[] for group in groups}

    #load word embeddings and stop words
    print("loading word embeddings...")
    word2vec = pickle.load( open( word2vec_path, "rb" ) )
    print("embeddings loaded")
    stop_words = get_stop_words(stop_words_path)

    #split the prediction data into groups
    predictions = read_predictions(predictions_path)
    for prediction in predictions:
        predicted_prob = float(prediction[0])
        real_class = int(prediction[1])
        predicted_class = int(prediction[2][0])
        if predicted_prob > thresholds[1] and predicted_prob < thresholds[2]:
            group_to_data['mid'].append(prediction)
        elif predicted_prob < thresholds[0]:
            if real_class == predicted_class:
                group_to_data['i_correct'].append(prediction)
            else:
                group_to_data['p_incorrect'].append(prediction)
        elif predicted_prob > thresholds[3]:
            if real_class == predicted_class:
                group_to_data['p_correct'].append(prediction)
            else:
                group_to_data['i_incorrect'].append(prediction)

    #output the groups into files for inspection
    for group in groups:
        output_path = group_to_output_path[group]
        writer = open(output_path, 'w')
        writer.write("predicted prob,true label,predicted label,sentence\n")
        for data in group_to_data[group]:
            writer.write(data[0]+','+data[1]+','+data[2][0]+','+data[3]+'\n')
        writer.close()
    print("categories printed")

    save_obj(group_to_data, group_to_data_path)

    plot_dict = get_plot_vectors(group_to_data, word2vec, vec_length, stop_words)
    print("tsne generated.")

    groups = list(plot_dict.keys())
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    fig, ax = plt.subplots()

    for i in range(len(groups)):
        group = groups[i]
        color = colors[i]
        marker = markers[i]
        labels = plot_dict[group][0]
        x = plot_dict[group][1]
        y = plot_dict[group][2]
        print(group, color)
        size = 0.5
        ax.scatter(x, y, color=color, marker=marker, s=size, label=group)
        ax.set_title("Selection of Sentences from Israeli and Palestinian Narratives", fontsize=6.5)
        plt.axis('off')
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]), fontsize = 0.5)

    plt.legend(prop={'size': 4})
    plt.savefig(tsne_output_path, dpi=900)
    print('image saved in', tsne_output_path)










