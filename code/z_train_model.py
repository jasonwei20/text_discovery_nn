## Usages:
## training the model with either word embeddings or bag of words, with either naive bayes, svm, random forest, logistic regression, or neural networks
## python code/3_train_model.py --format=embeddings --classifier=net
## arguments:
    ## format: embeddings, bag
    ## classifier: bayes, forest, svm, logistic, net

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

# for the bag, build a x_data and y_data from the lines, etc.
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
    x_data, y_data = sklearn.utils.shuffle(x_data, y_data, random_state=0)
    print("x_data.shape:", x_data.shape)
    print("y_data.shape:", y_data.shape)

    return x_data, y_data

def get_x_y_data_bag(isr, pal, word2idx, stop_words):

    x_data = np.zeros((len(isr)+len(pal), len(word2idx)))
    y_data = np.zeros((len(isr)+len(pal), 1))

    for i in range(len(isr)):
        line = isr[i]
        words = line[:-1].split(' ')
        for word in words:
            if word not in stop_words:
                x_data[i, word2idx[word]] = 1

    for i in range(len(pal)):
        line = pal[i]
        words = line[:-1].split(' ')
        for word in words:
            if word not in stop_words:
                x_data[len(isr)+i, word2idx[word]] = 1
        y_data[len(isr)+i] = 1

    x_data, y_data = sklearn.utils.shuffle(x_data, y_data, random_state=0)
    print("x_data.shape:", x_data.shape)
    print("y_data.shape:", y_data.shape)

    return x_data, y_data


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
    isr_val = isr_full[-200:-100] #validation set
    pal_val = pal_full[-200:-100]
    isr_test = isr_full[-100:] #testing set
    pal_test = pal_full[-100:]

    #normalize distribution of labels (there are more palestinian lines)
    isr_train = isr_train + isr_train + isr_train #training set
    pal_train = pal_train + pal_train
    pal_train = pal_train[:len(isr_train)] 
    print("data distribution for training set normalized to", len(isr_train), "and", len(pal_train))

    #variables that we care about
    x_data_full, y_data_full = None, None
    x_data_train, y_data_train = None, None
    x_data_val, y_data_val = None, None
    x_data_test, y_data_test = None, None
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
        x_data_test, y_data_test = get_x_y_data_embeddings(isr_test, pal_test, word2vec, stop_words)
    
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
        x_data_test, y_data_test = get_x_y_data_bag(isr_test, pal_test, word2idx, stop_words)
    
        
    print("line_to_vec and vec_to_line created with", len(line_to_vec), "pairs.")

    return x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, x_data_full, y_data_full, line_to_vec, vec_to_line


def train_model(x_data_train, y_data_train, classifier, format):

    assert classifier in ['bayes', 'forest', 'svm', 'logistic', 'net']
    feature_size = x_data_train.shape[1]

    model = None

    #naive bayes
    if classifier == 'bayes':
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(x_data_train, y_data_train)
        model = gnb

    #random forest
    if classifier == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        forest_clf = RandomForestClassifier(max_depth=5, random_state=0)
        forest_clf.fit(x_data_train, y_data_train)
        model = forest_clf

    #logistic regression
    if classifier == 'logistic':
        from sklearn import linear_model
        model = linear_model.LogisticRegression()
        model.fit(x_data_train, y_data_train)

    #two layer neural network in keras
    if classifier == 'net':

        # if format == 'embeddings':
        #     nb_epochs = 200
        #     dropout_rate = 0.9
        # elif format == 'bag':
        #     nb_epochs = 50
        #     dropout_rate = 0.95

        model = Sequential()
        model.add(Dense(64, input_dim=feature_size))
        model.add(Dense(32))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16))
        model.add(Dense(4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # model = Sequential()
        # model.add(Dense(128, input_dim=feature_size))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(64))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(32))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(16))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())


        model.fit(x_data_train, y_data_train, batch_size=128, nb_epoch=30, validation_split=0.05)

    return model

def evaluate_accuracy(model, x_data, y_data, classifier, format, set_type):

    y_predict_probs = get_predictions_probabilities(model, x_data, classifier)

    y_predict_classes = to_binary(y_predict_probs)
    print(set_type, "set accuracy:", accuracy_score(y_data, y_predict_classes))

def output_predictions(model, x_data_full, y_data_full, analysis_path, classifier, format):

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
    
    print("output each sentence with its true label, predicted label, and predicted probability in", analysis_path)




def get_predictions_probabilities(model, x_data, classifier):

    y_predict_probs = None
    if classifier == 'net':
        y_predict_probs = model.predict(x_data)
    else:
        y_predict_probs = model.predict_proba(x_data)[:, 1]
        y_predict_probs = np.expand_dims(y_predict_probs, axis=1)
    return y_predict_probs

def evaluate_combined_accuracy(model_b, model_e, x_data_b, x_data_e, y_data, classifier_b, classifier_e, format, set_type):

    y_predict_probs_b = get_predictions_probabilities(model_b, x_data_b, classifier_b)
    y_predict_probs_e = get_predictions_probabilities(model_e, x_data_e, classifier_e)
    y_predict_probs = ( y_predict_probs_b + y_predict_probs_e ) / 2

    y_predict_classes = to_binary(y_predict_probs)
    print(set_type, "set accuracy:", accuracy_score(y_data, y_predict_classes))

def output_combined_predictions(model_b, model_e, x_data_full_b, x_data_full_e, y_data_full, analysis_path, classifier_b, classifier_e, format, vec_to_line):

    #test the neural net on the test set
    y_predict_probs_b = get_predictions_probabilities(model_b, x_data_full_b, classifier_b)
    y_predict_probs_e = get_predictions_probabilities(model_e, x_data_full_e, classifier_e)
    y_predict_probs = ( y_predict_probs_b + y_predict_probs_e ) / 2

    y_predict_classes = to_binary(y_predict_probs)
    print("final full set accuracy:", accuracy_score(y_data_full, y_predict_classes))

    #sort and output each sentence with its true label, predicted label, and predicted probability
    predicted_prob_to_data = {}
    for i in range(x_data_full_b.shape[0]):
        vec = x_data_full_b[i]
        if format == 'embeddings':
            sentence = vec_to_line[str(vec)]
        elif format == 'bag':
            sentence = vec_to_line["".join(map(str, vec.tolist()))]
        true_label = y_data_full[i][0]
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
    
    print("output each sentence with its true label, predicted label, and predicted probability in", analysis_path)


def get_prediction_distribution(y_predict_probs, bins):

    bin_size = 1.0/bins
    bin_to_num = {b:0 for b in range(bins)}

    for prob in np.squeeze(y_predict_probs).tolist():
        bin_to_num[int(prob/bin_size)] += 1
    for _bin in bin_to_num:
        bin_to_num[_bin] = bin_to_num[_bin]/y_predict_probs.shape[0]
    return bin_to_num

def get_dist_differential(model, x_data, classifier):

    y_predict_probs = get_predictions_probabilities(model, x_data, classifier)
    bin_to_num = get_prediction_distribution(y_predict_probs, 100)

    loss = 0
    for _bin in bin_to_num:
        added_loss = abs(bin_to_num[_bin] - 1.0/len(bin_to_num)) / len(bin_to_num)
        loss += added_loss

    return loss


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
    analysis_path = "outputs/predictions_"+args.format+"_"+args.classifier+".csv"

    #load dataset
    x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, x_data_full, y_data_full, line_to_vec, vec_to_line = load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path, args.format)

    model = train_model(x_data_train, y_data_train, args.classifier, args.format)

    evaluate_accuracy(model, x_data_train, y_data_train, args.classifier, args.format, "train")
    evaluate_accuracy(model, x_data_val, y_data_val, args.classifier, args.format, "val")
    evaluate_accuracy(model, x_data_test, y_data_test, args.classifier, args.format, "test")
    output_predictions(model, x_data_full, y_data_full, analysis_path, args.classifier, args.format)
    loss = get_dist_differential(model, x_data_full, args.classifier)
    print(loss)
    #initialize small neural net in keras



    #train neural net
    











