# text_discovery_nn

This project uses word embeddings and neural networks to discover potentially significant segments of text in historical narratives. The dataset I used for this is the book [Side by Side](https://www.amazon.com/Side-Parallel-Histories-Israel-Palestine/dp/1595586830/ref=sr_1_1?ie=UTF8&qid=1532494633&sr=8-1&keywords=side+by+side+sami+adwan) by Sami Adwan, which tells the history of the Israeli Palestine conflict from both Israeli and Palestinian viewpoints, on pages side-by-side. This code automatically discovers the most contended events in this fascinating narrative.

Dependencies:
- SKLearn
- Keras with tensorflow backend
- Matplotlib
- 200d [word embedding vectors](https://nlp.stanford.edu/projects/glove/) from twitter. 

Data:
- My data are the 626 paragraphs from the Israeli narrative and 891 paragraphs from the Palestinian narrative.
- All punctuation and stop words are removed.

Methods:
First, label all paragraphs from the Israeli narrative as '0' and the Palestinian narrative as '1'. Then train a neural net to predict either Israeli or Palestinian bias from the average word embeddings of each paragraph. After convergence, rerun on all training data to get a predicted score from 0-1, with 0 leaning towards Israeli bias and 1 leaning towards Palestinian bias. Then, bucket the sentences that fall into one of the five categories (note, some fall into none of the five):

| Category | True Label | Predicted score  |
| ------------- |:-------------:|:-----:|
| Israeli-biased | Israeli | < 0.2 |
| Israeli-sympathetic      | Israeli     |  > 0.8 |
| Palestinian-biased | Palestinian     |  > 0.8 |
| Palestinian-sympathetic | Palestinian     |  < 0.2 |
| Neutral | either    |   between 0.4 and 0.6 |

Israeli-biased means the model strongly predicted Israeli bias in a certain paragraph. Israeli-symphathetic means the model strongly predicted Palestinian bias, indicating that the Israeli narrative may be more sympathetic to the other side for that particular event. Etc.

I then cluster the sentences by the following: sentences were added to a cluster only if they were in the same one of the five categories as another sentence no more than two sentences away in the chronology of the book.

Future work:
- Thresholds were chosen arbitrarily and could be tuned.

## Miscellaneous Code

Creating word to index, index to word, and word to vec dictionaries from a file:
```
python code/nlp_utils.py --function=gen_vocab_dicts
```





