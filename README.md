# Historical Text Discovery with Machine Learning

This project uses word embeddings and neural networks to discover potentially significant segments of text in historical narratives. The dataset I used for this is the book [Side by Side](https://www.amazon.com/Side-Parallel-Histories-Israel-Palestine/dp/1595586830/ref=sr_1_1?ie=UTF8&qid=1532494633&sr=8-1&keywords=side+by+side+sami+adwan) by Sami Adwan, which tells the history of the Israeli Palestine conflict from both Israeli and Palestinian viewpoints, on pages side-by-side. This code automatically discovers the most contended events in this fascinating narrative.

### Dependencies:
- SKLearn
- Keras with tensorflow backend
- Matplotlib
- 300d [word embedding vectors](https://nlp.stanford.edu/projects/glove/) from common crawl.

### Data:
We use the Israeli and Palestinian historical narratives compiled in Side by Side by Sami Adwan as our dataset. The book was purchased in print, and each page was separated (labeled) based on the origin. The book was then sent to a third-party book scanning company for conversion into electronic format. I assign texts from the Israeli narrative the label $0$ and texts from the Palestinian narrative $1$. Of note, 731 of the 8214 letter sequences (about 9\%) in our text corpus were not found in the Common Crawl pre-trained embeddings, indicating a large amount of noise in the conversion of the book from hard-copy to digital. 

### Method Overview and Historical Analysis
A text classification model such as a random forest, logistic regression, or feedforward neural net is trained on the labeled dataset described in the previous subsection. After we optimize the model with classifier selection and hyperparameter tuning, we then rerun the model over the entire training set, and get some predicted score for each article. In typically classification tasks, this is done on new data, and an inference is made based on whether the predicted score is greater than or less than 0.5. However, here we use the predicted score to rank each article, and then deem some articles as "notable" based on their predicted confidence and true label. The following figure shows a schematic for this training and evaluation process, and the following shows the five notable categories. Not all texts are sorted into one of the five categories.

| Predicted Score | True label: Israeli | True label: Palestinian  |
| ------------- |:-------------:|:-----:|
| (0, 0.25) | Israeli-biased | Palestinian-sympathetic |
| (0.4, 0.6)     | Neutral     |  Neutral  |
| (0.75, 1) | Israeli-sympathetic     |  Palestinian-sympathetic |

![hello](https://github.com/jasonwei20/text_discovery_nn/blob/master/selected-outputs/methods.png)

Method overview schematic. The model is trained on a labeled text corpus, and then re-classifies the training set to get some confidence score for each article. Blue outlines indicate Israeli sources, while red outlines indicate Palestinian sources. Then, misclassifications are labeled into five categories: Israeli-biased, Palestinian-biased, Israeli-sympathetic, Palestinian-sympathetic, and Neutral. Israeli-sympathetic means that the model predicted high Palestinian-bias for an article with a true label of Israeli, indicating that this may have been an event for which the Israeli's were sympathetic to the Palestinian side.

### Ablation Test

To choose the best classifier, I performed an ablation test with a number of classifiers, and chose the one with the best combination of validation accuracy and maximum entropy. I trained and tested the naive bayes, random forest, logistic regression, and neural network classifiers on the same normalized training set, using both word embeddings and bag of words representations.  I use n=30 to get some entropy score gamma for each classifier, in addition to measuring its validation accuracy on a hold-out set of 200 samples per class. The below figure shows the validation accuracy and entropy of all four classifiers, where I scale entropy to log(1/gamma), since gamma is the area between the predicted distribution and the uniform, so it thus measures the inverse of entropy. 

![hello](https://github.com/jasonwei20/text_discovery_nn/blob/master/selected-outputs/ablation_test.png)
Results of ablation test on four different classifiers and two different numerical representations of articles. Green denotes use of bag of words representations, and blue denotes use of word embeddings. 'nb' = naive bayes, 'rf' = random forest, 'lr' = logistic regression, and 'nn' = neural network.

To further test my trained classifier to see if it works, I manually collected more data for an external set of different origin to see whether my classifier is able to detect bias. I googled for biased articles of the Israeli-Palestine conflict, and I found one website on algeminer.com [[link]](https://www.algemeiner.com/2017/05/28/new-york-times-unleashes-onslaught-of-five-op-eds-hostile-to-israel/)  titled "New York Times Unleashes Onslaught of Five Op-eds hostile to Israel." This seems like a pretty promising source of Palestinian-biased articles (or at least from the Palestinian narrative), so I went ahead and downloaded them, cleaned them, and put them through the classifier. My classifier detected an average score of 0.534, with 37 articles of Israeli bias and 47 articles of Palestinian bias, which is overall slightly Palestinian-biased, the result we are looking for. However, this is not as strong as I would have hoped for. I manually examined some of the predicted Israeli-biased articles with high confidence, and it seems that these are often related to President Trump or Iran's nuclear development and foreign policy. The top 5 most confidence results for Palestinian-bias and Israeli-bias, with their confidence scores, are shown in the the tsv file in `selected outputs`.








