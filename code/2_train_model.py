## Usages:
## training the model with either word embeddings or bag of words, with either naive bayes, svm, random forest, logistic regression, or neural networks
## python code/3_train_model.py --format=embeddings --classifier=net
## arguments:
    ## format: embeddings, bag
    ## classifier: bayes, forest, svm, logistic, net

## inputs: vocab dictionaries, stop words, israeli lines, palestinian lines, prediction output path


## Jason Wei
## September 10, 2018
## jason.20@dartmouth.edu

from nlp_utils import *
np.set_printoptions(threshold=np.nan)
import random
random.seed(42)
from numpy.random import seed
seed(42)

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
    x_data, y_data = sklearn.utils.shuffle(x_data, y_data, random_state=42)
    print("x_data.shape:", x_data.shape)
    print("y_data.shape:", y_data.shape)

    return x_data, y_data

#load the data
def load_all_data(vocab_dicts_path, isr_path, pal_path, stop_words_path):

    #load everything from the path
    word2idx, idx2word, word2vec = pickle.load(open(vocab_dicts_path, 'rb'))
    isr = open(isr_path, 'r').readlines()
    isr = [line.split('\t')[-1] for line in isr]
    pal = open(pal_path, 'r').readlines() 
    pal = [line.split('\t')[-1] for line in pal]
    stop_words = get_stop_words(stop_words_path)

    line_to_vec, vec_to_line = {}, {}

    # get the line to vec and vec to line
    all_lines = isr + pal
    for line in all_lines:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        line_to_vec[line] = avg_vec
        vec_to_line[str(avg_vec)] = line

    x_data, y_data = get_x_y_data_embeddings(isr, pal, word2vec, stop_words)
    print("line_to_vec and vec_to_line created with", len(line_to_vec), "pairs.")

    return x_data, y_data, line_to_vec, vec_to_line


#load the data
def load_train_data(vocab_dicts_path, isr_path, pal_path, stop_words_path):

    #load everything from the path
    word2idx, idx2word, word2vec = pickle.load(open(vocab_dicts_path, 'rb'))
    isr = open(isr_path, 'r').readlines() #full data set
    pal = open(pal_path, 'r').readlines() 
    stop_words = get_stop_words(stop_words_path)

    isr = isr + isr
    isr = isr[:len(pal)] 

    line_to_vec, vec_to_line = {}, {}

    # get the line to vec and vec to line
    all_lines = isr + pal
    for line in all_lines:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        line_to_vec[line] = avg_vec
        vec_to_line[str(avg_vec)] = line

    x_data, y_data = get_x_y_data_embeddings(isr, pal, word2vec, stop_words)
    print("line_to_vec and vec_to_line created with", len(line_to_vec), "pairs.")

    return x_data, y_data, line_to_vec, vec_to_line


#load the data
def load_data(vocab_dicts_path, isr_path, pal_path, stop_words_path):

    #load everything from the path
    word2idx, idx2word, word2vec = pickle.load(open(vocab_dicts_path, 'rb'))
    isr = open(isr_path, 'r').readlines() #full data set
    pal = open(pal_path, 'r').readlines() 
    stop_words = get_stop_words(stop_words_path)

    line_to_vec, vec_to_line = {}, {}

    # get the line to vec and vec to line
    all_lines = isr + pal
    for line in all_lines:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        line_to_vec[line] = avg_vec
        vec_to_line[str(avg_vec)] = line

    x_data, y_data = get_x_y_data_embeddings(isr, pal, word2vec, stop_words)
    print("line_to_vec and vec_to_line created with", len(line_to_vec), "pairs.")

    return x_data, y_data, line_to_vec, vec_to_line



def train_model(x_data_train, y_data_train, classifier):

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
        forest_clf = RandomForestClassifier(max_depth=10, random_state=0)
        forest_clf.fit(x_data_train, y_data_train)
        model = forest_clf

    #logistic regression
    if classifier == 'logistic':
        from sklearn import linear_model
        model = linear_model.LogisticRegression()
        model.fit(x_data_train, y_data_train)

    #two layer neural network in keras
    if classifier == 'net':

        nb_epochs = 30
        dropout_rate = 0.2

        model = Sequential()
        model.add(Dense(64, input_dim=feature_size))
        model.add(Dense(32))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16))
        model.add(Dense(4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_data_train, y_data_train, batch_size=128, nb_epoch=nb_epochs, validation_split=0.05)

    return model


def get_predictions_probabilities(model, x_data, classifier):

    y_predict_probs = None
    if classifier == 'net':
        y_predict_probs = model.predict(x_data)
    else:
        y_predict_probs = model.predict_proba(x_data)[:, 1]
        y_predict_probs = np.expand_dims(y_predict_probs, axis=1)
    return y_predict_probs

def evaluate_accuracy(model, x_data, y_data, classifier, set_type):

    y_predict_probs = get_predictions_probabilities(model, x_data, classifier)
    y_predict_classes = to_binary(y_predict_probs)
    print(set_type, "set accuracy:", accuracy_score(y_data, y_predict_classes))

def output_predictions(model, x_data, y_data, analysis_path, classifier):

    #test the neural net on the test set
    y_predict_probs = None
    if classifier == 'net':
        y_predict_probs = model.predict(x_data)

    else:
        y_predict_probs = model.predict_proba(x_data)[:, 1]
        y_predict_probs = np.expand_dims(y_predict_probs, axis=1)

    y_predict_classes = to_binary(y_predict_probs)
    print("final full set accuracy:", accuracy_score(y_data, y_predict_classes))

    #sort and output each sentence with its true label, predicted label, and predicted probability
    predicted_prob_to_data = {}
    for i in range(x_data.shape[0]):
        vec = x_data[i]
        sentence = vec_to_line_all[str(vec)]
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

def get_prediction_distribution(y_predict_probs, bins):

    bin_size = 1.0/bins
    bin_to_num = {b:0 for b in range(bins)}

    for prob in np.squeeze(y_predict_probs).tolist():
        if prob == 1.0:
            prob = 0.999
        bin_to_num[int(prob/bin_size)] += 1
    for _bin in bin_to_num:
        bin_to_num[_bin] = bin_to_num[_bin]/y_predict_probs.shape[0]
    return bin_to_num

def get_dist_differential(model, x_data, classifier):

    y_predict_probs = get_predictions_probabilities(model, x_data, classifier)
    bin_to_num = get_prediction_distribution(y_predict_probs, 30)

    loss = 0
    for _bin in bin_to_num:
        added_loss = abs(bin_to_num[_bin] - 1.0/len(bin_to_num)) / len(bin_to_num)
        loss += added_loss

    return loss

def get_predictions_probabilities(model, x_data, classifier):

    y_predict_probs = None
    if classifier == 'net':
        y_predict_probs = model.predict(x_data)
    else:
        y_predict_probs = model.predict_proba(x_data)[:, 1]
        y_predict_probs = np.expand_dims(y_predict_probs, axis=1)
    return y_predict_probs

def plot_histo_distribution(model, x_data, classifier, num_bins):

    y_predict_probs = get_predictions_probabilities(model, x_data, classifier)
    fig, ax = plt.subplots()
    ax.hist(y_predict_probs, num_bins)
    ax.set_title("Distribution of Predicted Scores", fontsize=12)
    plt.xlabel("Score")
    plt.ylabel('Number of Samples')
    plt.savefig("outputs/distribution.png", dpi=400)



if __name__ == "__main__":    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str)
    args = parser.parse_args()

    #input parameters
    vocab_dicts_path = "pickles/vocab_dicts.p"
    stop_words_path = "processed_data/stop_words.txt"
    analysis_path = "outputs/predictions_"+args.classifier+".csv"

    x_all, y_all, line_to_vec_all, vec_to_line_all = load_all_data(vocab_dicts_path, "processed_data/nn/i_all.txt", "processed_data/nn/p_all.txt", stop_words_path)
    x_train, y_train, line_to_vec_train, vec_to_line_train = load_train_data(vocab_dicts_path, "processed_data/nn/i_train.txt", "processed_data/nn/p_train.txt", stop_words_path)
    x_dev, y_dev, line_to_vec_dev, vec_to_line_dev = load_data(vocab_dicts_path, "processed_data/nn/i_dev.txt", "processed_data/nn/p_dev.txt", stop_words_path)
    x_test, y_test, line_to_vec_test, vec_to_line_test = load_data(vocab_dicts_path, "processed_data/nn/i_test.txt", "processed_data/nn/p_test.txt", stop_words_path)
    print(x_test.shape, y_test.shape)

    model = train_model(x_train, y_train, args.classifier)
    evaluate_accuracy(model, x_train, y_train, args.classifier, "train")
    evaluate_accuracy(model, x_dev, y_dev, args.classifier, "val")
    evaluate_accuracy(model, x_test, y_test, args.classifier, "test")
    evaluate_accuracy(model, x_all, y_all, args.classifier, "all")
    output_predictions(model, x_all, y_all, analysis_path, args.classifier)
    plot_histo_distribution(model, x_all, args.classifier, 50)
    loss = get_dist_differential(model, x_all, args.classifier)
    print(loss)

    vec_dicts = (line_to_vec_all, vec_to_line_all)
    pickle.dump( vec_dicts, open( "vec_dicts.p", "wb" ) )



