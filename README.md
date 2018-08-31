# text_discovery_nn

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
| (0, 0.25) | Israeli--biased | Palestinian--sympathetic |
| (0.4, 0.6)     | Neutral     |  Neutral  |
| (0.75, 1) | Israeli--sympathetic     |  Palestinian--sympathetic |




