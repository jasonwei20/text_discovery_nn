## Usages:
## training the model with either word embeddings or bag of words, with either naive bayes, svm, random forest, logistic regression, or neural networks
## python code/2_grid_search.py
## arguments:

## inputs: vocab dictionaries, stop words, israeli lines, palestinian lines, prediction output path


## Jason Wei
## August 1, 2018
## jason.20@dartmouth.edu

from nlp_utils import *
np.set_printoptions(threshold=np.nan)
import random
random.seed(42)
from numpy.random import seed
seed(42)
from sklearn.utils import shuffle
import operator

#get the embeddings representation of data
def get_x_y_data_embeddings(isr, pal, word2vec, stop_words):

    x_data, y_data = [], []

    for line in isr:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        x_data.append(avg_vec)
        y_data.append([0])

    for line in pal:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        x_data.append(avg_vec)
        y_data.append([1])

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    print("x_data.shape:", x_data.shape)
    print("y_data.shape:", y_data.shape)

    return x_data, y_data

#get the bag of words representation of data
def get_x_y_data_bag(isr, pal, word2idx, stop_words):

    x_data = np.zeros((len(isr)+len(pal), len(word2idx)))
    y_data = np.zeros((len(isr)+len(pal), 1))

    for i in range(len(isr)):
        line = isr[i]
        words = line[:-1].split(' ')
        for word in words:
            if word not in stop_words:
                x_data[i, word2idx[word]] = 0.01

    for i in range(len(pal)):
        line = pal[i]
        words = line[:-1].split(' ')
        for word in words:
            if word not in stop_words:
                x_data[len(isr)+i, word2idx[word]] = 0.01
        y_data[len(isr)+i] = 1

    x_data, y_data = shuffle(x_data, y_data, random_state=42)
    print("x_data.shape:", x_data.shape)
    print("y_data.shape:", y_data.shape)

    return x_data, y_data

#load the data
def load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, format):

    assert format in ['embeddings', 'bag']

    #load everything from the path
    word2idx, idx2word, word2vec = pickle.load(open(vocab_dicts_path, 'rb'))
    isr_full = open(isr_path, 'r').readlines() #full data set
    pal_full = open(pal_path, 'r').readlines() 
    print(len(isr_full), "israeli lines loaded,", len(pal_full), "palestinian lines loaded")
    stop_words = get_stop_words(stop_words_path)

    shuffle(isr_full)
    shuffle(pal_full)
    isr_train = isr_full[:-200] 
    pal_train = pal_full[:-200] 
    isr_val = isr_full[-200:] #validation set
    pal_val = pal_full[-200:]

    #normalize distribution of labels (there are more palestinian lines)
    isr_train = isr_train + isr_train + isr_train #training set
    pal_train = pal_train + pal_train + pal_train
    pal_train = pal_train[:len(isr_train)] 
    print("data distribution for training set normalized to", len(isr_train), "and", len(pal_train))

    #variables that we care about
    x_data_full, y_data_full = None, None
    x_data_train, y_data_train = None, None
    x_data_val, y_data_val = None, None
    line_to_vec, vec_to_line = {}, {}

    if format == 'embeddings':

        # get the line to vec and vec to line
        all_lines = isr_full + pal_full
        for line in all_lines:
            avg_vec = get_avg_vec(line, word2vec, stop_words)
            line_to_vec[line] = avg_vec
            vec_to_line[str(avg_vec)] = line
    
        x_data_full, y_data_full = get_x_y_data_embeddings(isr_full, pal_full, word2vec, stop_words)
        x_data_train, y_data_train = get_x_y_data_embeddings(isr_train, pal_train, word2vec, stop_words)
        x_data_val, y_data_val = get_x_y_data_embeddings(isr_val, pal_val, word2vec, stop_words)
    
    if format == 'bag':
        
        # get the line to vec and vec to line
        all_lines = isr_full + pal_full
        x_data_temp = np.zeros((len(isr_full)+len(pal_full), len(word2idx)))
        for i in range(len(all_lines)):
            line = all_lines[i]
            words = line[:-1].split(' ')
            for word in words:
                if word not in stop_words:
                    x_data_temp[i, word2idx[word]] = 1
            line_to_vec[line] = x_data_temp[i]
            vec_to_line["".join(map(str, x_data_temp[i].tolist()))] = line

        x_data_full, y_data_full = get_x_y_data_bag(isr_full, pal_full, word2idx, stop_words)
        x_data_train, y_data_train = get_x_y_data_bag(isr_train, pal_train, word2idx, stop_words)
        x_data_val, y_data_val = get_x_y_data_bag(isr_val, pal_val, word2idx, stop_words)    
        
    print("line_to_vec and vec_to_line created with", len(line_to_vec), "pairs.")

    return x_data_train, y_data_train, x_data_val, y_data_val, x_data_full, y_data_full, line_to_vec, vec_to_line


def train_net(x_data_train, y_data_train, num_epochs, dropout_rate):

    feature_size = x_data_train.shape[1]

    model = Sequential()
    model.add(Dense(64, input_dim=feature_size))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    start = time.time()
    model.fit(x_data_train, y_data_train, batch_size=128, nb_epoch=num_epochs, validation_split=0.05, verbose=0)
    return model

def evaluate_accuracy(model, x_data, y_data, set_type):

    y_predict_probs = model.predict(x_data)
    y_predict_classes = to_binary(y_predict_probs)
    accuracy = accuracy_score(y_data, y_predict_classes)
    print(set_type, "set accuracy:", accuracy)
    return accuracy

def get_output_predictions(model, x_data_full):

    y_predict_probs = model.predict(x_data_full)
    return y_predict_probs

def get_prediction_distribution(y_predict_probs, bins):

    bin_size = 1.0/bins
    bin_to_num = {b:0 for b in range(bins)}

    for prob in np.squeeze(y_predict_probs).tolist():
        bin_to_num[int(prob/bin_size)] += 1
    for _bin in bin_to_num:
        bin_to_num[_bin] = bin_to_num[_bin]/y_predict_probs.shape[0]
    return bin_to_num

def get_dist_differential(model, x_data_full):

    y_predict_probs = get_output_predictions(model, x_data_full)
    bin_to_num = get_prediction_distribution(y_predict_probs, 100)

    loss = 0
    for _bin in bin_to_num:
        added_loss = abs(bin_to_num[_bin] - 1.0/len(bin_to_num)) / len(bin_to_num)
        loss += added_loss

    return loss

def output_predictions(model, x_data, y_data, analysis_path, classifier, format):

    #test the neural net on the test set
    y_predict_probs = None
    if classifier == 'net':
        y_predict_probs = model.predict(x_data_full)

    else:
        y_predict_probs = model.predict_proba(x_data_full)[:, 1]
        y_predict_probs = np.expand_dims(y_predict_probs, axis=1)

    y_predict_classes = to_binary(y_predict_probs)
    print("final full set accuracy:", accuracy_score(y_data_full, y_predict_classes))

    #sort and output each sentence with its true label, predicted label, and predicted probability
    predicted_prob_to_data = {}
    for i in range(x_data.shape[0]):
        vec = x_data[i]
        if format == 'embeddings':
            sentence = vec_to_line[str(vec)]
        elif format == 'bag':
            sentence = vec_to_line["".join(map(str, vec.tolist()))]
        true_label = y_data[i][0]
        predicted_label = y_predict_classes[i][0]
        predicted_prob = y_predict_probs[i][0]
        predicted_prob_to_data[predicted_prob] = (true_label, predicted_label, sentence)
    writer = open(analysis_path, 'w')
    writer.write("predicted prob,true label,predicted label,sentence\n")
    for predicted_prob in sorted(predicted_prob_to_data):
        true_label = predicted_prob_to_data[predicted_prob][0]
        predicted_label = predicted_prob_to_data[predicted_prob][1]
        sentence = predicted_prob_to_data[predicted_prob][2]
        writer.write("{:.7f}".format(predicted_prob) + "," + str(true_label) + "," + str(predicted_label) + "," + str(sentence))
    writer.close()



def get_stacked_data(vocab_dicts_path, isr_path, pal_path, stop_words_path):

    #get the bag of words and word embedding formats
    x_data_train_b, y_data_train_b, x_data_val_b, y_data_val_b, x_data_full_b, y_data_full_b, line_to_vec_b, vec_to_line_b = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, "bag")
    x_data_train_e, y_data_train_e, x_data_val_e, y_data_val_e, x_data_full_e, y_data_full_e, line_to_vec_e, vec_to_line_e = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, "embeddings")

    #stack the two to get the input vectors
    x_data_train_s = np.concatenate((x_data_train_b, x_data_train_e), axis=1)
    x_data_val_s = np.concatenate((x_data_val_b, x_data_val_e), axis=1)
    x_data_full_s = np.concatenate((x_data_full_b, x_data_full_e), axis=1)
    print(x_data_train_s.shape, x_data_val_s.shape, x_data_full_s.shape)

    return x_data_train_s, x_data_val_s, x_data_full_s, y_data_train_e, y_data_val_e, y_data_full_e


if __name__ == "__main__":    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str)
    parser.add_argument("--classifier", type=str)
    args = parser.parse_args()

    #input parameters
    vocab_dicts_path = "pickles/vocab_dicts.p"
    isr_path = "processed_data/israeli_all.txt"
    pal_path = "processed_data/palestinian_all.txt"
    stop_words_path = "processed_data/stop_words.txt"
    analysis_path = "outputs/predictions_optimized_net.csv"

    #x_data_train, x_data_val, x_data_full, y_data_train, y_data_val, y_data_full = get_stacked_data(vocab_dicts_path, isr_path, pal_path, stop_words_path)
    x_data_train, y_data_train, x_data_val, y_data_val, x_data_full, y_data_full, line_to_vec, vec_to_line = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, "embeddings")

    #do a grid search over a bunch of paramters to get the best network
    epoch_and_dropout_to_score = {}
    test_epochs = [15]#, 100, 200, 300, 500]
    test_dropout_values = [0.9]#, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
    for dropout_value in test_dropout_values:
        for num_epoch in test_epochs:
            print("training for", num_epoch, dropout_value)
            model = train_net(x_data_train, y_data_train, num_epoch, dropout_value)
            train_accuracy = evaluate_accuracy(model, x_data_train, y_data_train, "train")
            val_accuracy = evaluate_accuracy(model, x_data_val, y_data_val, "val")
            dist_diff = get_dist_differential(model, x_data_full)
            print(dist_diff)
            score = val_accuracy - (train_accuracy - val_accuracy)*2/3
            epoch_and_dropout_to_score[(num_epoch, dropout_value)] = score
            print()
            output_predictions(model, x_data_full, y_data_full, analysis_path, args.classifier, args.format)

    
    sorted_x = sorted(epoch_and_dropout_to_score.items(), key=operator.itemgetter(1))
    for tup in sorted_x:
    	print(tup)



    # if args.format == "ensemble":

    #     #load dataset
    #     x_data_train_b, y_data_train_b, x_data_val_b, y_data_val_b, x_data_test_b, y_data_test_b, x_data_full_b, y_data_full_b, line_to_vec_b, vec_to_line_b = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, "bag")
    #     x_data_train_e, y_data_train_e, x_data_val_e, y_data_val_e, x_data_test_e, y_data_test_e, x_data_full_e, y_data_full_e, line_to_vec_e, vec_to_line_e = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, "embeddings")

    #     model_b, model_e = None, None
    #     classifier_1, classifier_2 = None, None

    #     if args.classifier == "both":
    #         classifier_1 = "net"
    #         classifier_2 = "logistic"
    #         model_b = train_model(x_data_train_b, y_data_train_b, "net", "bag")
    #         model_e = train_model(x_data_train_e, y_data_train_e, "logistic", "embeddings")

    #     else:
    #         classifier_1 = args.classifier
    #         classifier_2 = args.classifier
    #         model_b = train_model(x_data_train_b, y_data_train_b, args.classifier, "bag")
    #         model_e = train_model(x_data_train_e, y_data_train_e, args.classifier, "embeddings")

    #     evaluate_combined_accuracy(model_b, model_e, x_data_train_b, x_data_train_e, y_data_train_e, classifier_1, classifier_2, args.format, "train")
    #     evaluate_combined_accuracy(model_b, model_e, x_data_val_b, x_data_val_e, y_data_val_e, classifier_1, classifier_2, args.format, "val")
    #     evaluate_combined_accuracy(model_b, model_e, x_data_test_b, x_data_test_e, y_data_test_e, classifier_1, classifier_2, args.format, "test")
    #     output_combined_predictions(model_b, model_e, x_data_full_b, x_data_full_e, y_data_full_e, analysis_path, classifier_1, classifier_2, "bag", vec_to_line_b)

    # else:
    #     #load dataset
    #     x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, x_data_full, y_data_full, line_to_vec, vec_to_line = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, args.format)

    #     model = train_model(x_data_train, y_data_train, args.classifier, args.format)

    #     evaluate_accuracy(model, x_data_train, y_data_train, args.classifier, args.format, "train")
    #     evaluate_accuracy(model, x_data_val, y_data_val, args.classifier, args.format, "val")
    #     evaluate_accuracy(model, x_data_test, y_data_test, args.classifier, args.format, "test")
    #     #output_predictions(model, x_data_full, y_data_full, analysis_path, args.classifier, args.format)
    #     #initialize small neural net in keras



    #train neural net
    









