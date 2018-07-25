# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# Utility functions for NLP dataset.

import pickle
import sklearn
import keras
import numpy as np
from sklearn.utils import shuffle
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def find_line_number(requested_line, lines):
	for i in range(len(lines)):
		if requested_line == lines[i]:
			return i
	return None

def read_predictions(predictions_path):
	predictions_line = open(predictions_path, 'r').readlines()[1:]
	data = []
	for line in predictions_line:
		line_data = line.split(",")
		data.append(tuple(line_data))
	return data

def get_list_of_words(line):
    temp = line.replace('\n','').lower()
    temp = "".join(char for char in temp if char in 'qwertyuiopasdfghjklzxcvbnm ')
    return temp.split(' ')

def get_stop_words(file_path):
    stop_words_line = open(file_path, 'r').readlines()
    stop_words = [word[:-1] for word in stop_words_line]
    return stop_words

def remove_all_stop_words(list_of_words, stop_words):
    return [word for word in list_of_words if word not in stop_words]

def get_avg_vec(line, word2vec, stop_words):
    words = get_list_of_words(line)
    #words = remove_all_stop_words(words, stop_words)
    num_words = 0
    vec_length = len(word2vec['a'])
    avg = np.zeros((vec_length))
    
    for word in words:
        if word in word2vec:
            vec = word2vec[word]
            avg = np.add(avg, vec)
            num_words += 1
            
    if num_words > 0:
        avg = avg/num_words
    
    return avg

def to_binary(np_array):
	binary_array = np.zeros(np_array.shape)
	for i in range(np_array.shape[0]):
		if np_array[i] <= 0.5:
			binary_array[i] = 0
		else:
			binary_array[i] = 1
	return binary_array

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_plot_vectors(label_dictionary, word2vec, emb_size, stop_words):

    groups = label_dictionary.keys()

    #create embedding matrix
    lines = []
    for group in groups:
        line_groups = label_dictionary[group]
        lines += line_groups
    word_embeddings = np.zeros((len(lines), emb_size))
    for i in range(len(lines)):
        line = lines[i]
        word_embeddings[i, :] = get_avg_vec(line[3], word2vec, stop_words)

    #get the tsne for this
    tsne = TSNE(n_components=2).fit_transform(word_embeddings)
    
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











