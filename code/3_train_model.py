# Jason Wei
# July 24, 2018
# jason.20@dartmouth.edu

# Here, we discover historical bias in the Israeli-Palestine conflict with word embeddings.

from nlp_utils import *

if __name__ == "__main__":    

    #all the data sources
    vec_length = 200
    word2vec_path = "word2vec/word2vec_200.p"
    stop_words_path = "processed_data/stop_words.txt"
    isr_path = "processed_data/israeli_all.txt"
    pal_path = "processed_data/palestinian_all.txt"
    analysis_path = "outputs/nn_predictions.csv"

    #load word embeddings and stop words
    print("loading word embeddings...")
    word2vec = pickle.load( open( word2vec_path, "rb" ) )
    print("embeddings loaded")
    stop_words = get_stop_words(stop_words_path)

    #load dataset
    x_data, y_data = [], []
    line_to_vec, vec_to_line = {}, {}
    isr = open(isr_path, 'r').readlines()
    pal = open(pal_path, 'r').readlines()
    print(len(isr), "israeli lines loaded,", len(pal), "palestinian lines loaded")

    #label all israeli paragraphs as '0'
    for line in isr:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        x_data.append(avg_vec)
        line_to_vec[line] = avg_vec
        vec_to_line[str(avg_vec)] = line
        y_data.append([0])

    #label all palestinian paragraphs as '1'
    for line in pal:
        avg_vec = get_avg_vec(line, word2vec, stop_words)
        x_data.append(avg_vec)
        line_to_vec[line] = avg_vec
        vec_to_line[str(avg_vec)] = line
        y_data.append([1])

    #shuffle and split data
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    x_data_full, y_data_full = shuffle(x_data, y_data, random_state=0)
    x_data = x_data_full[:625]
    y_data = y_data_full[:625] #normalize the class distribution
    #x_data, X_test, y_data, y_test = train_test_split(x_data, y_data, test_size=0.05, random_state=42)
    print("x_data_shape:", x_data.shape)
    print("y_data_shape:", y_data.shape)

    #initialize small neural net in keras
    model = Sequential()
    model.add(Dense(80, input_dim=vec_length))
    model.add(Dense(10))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #train neural net
    start = time.time()
    model.fit(x_data, y_data, batch_size=32, nb_epoch=50, validation_split=0.05)
    print('training time : ', time.time() - start)

    #test the neural net on the test set
    y_predict_probs = model.predict(x_data_full)
    y_predict_classes = to_binary(y_predict_probs)
    print("final training accuracy:", accuracy_score(y_data_full, y_predict_classes))

    #sort and output each sentence with its true label, predicted label, and predicted probability
    predicted_prob_to_data = {}
    for i in range(x_data_full.shape[0]):
        vec = x_data_full[i]
        sentence = vec_to_line[str(vec)]
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
        writer.write("{:.5f}".format(predicted_prob) + "," + str(true_label) + "," + str(predicted_label) + "," + str(sentence))
    writer.close()
    
    print("output each sentence with its true label, predicted label, and predicted probability in", analysis_path)










