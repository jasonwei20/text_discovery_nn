import numpy as np
import pickle

file = open('glove.twitter.27B.200d.txt', 'r') 
text_embeddings = file.readlines()
word2vec = {}
for line in text_embeddings:
    items = line.split(' ')
    word = items[0]
    vec = items[1:]
    word2vec[word] = np.asarray(vec, dtype = 'float32')

pickle.dump(word2vec, open('word2vec_300.p', 'wb'))