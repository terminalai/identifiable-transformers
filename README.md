# Making Transformers Identifiable 

This repository helps someone:
* who is looking for a **quick** transformer-based classifier with low computation budget.
* wants to **tweak** the size of key vector and value vector, independently.
* wants to make their analysis of attention weights more **reliable**. How? see below...

### How to make your attention weights more reliable?

_As shown in our work (experimentally and theoretically)- for a given input X, a set of attention weights A, and output transformer prediction probabilities Y, if we can find another set of attention architecture generatable weights A* satisfying X-Y pair, analysis performed over A is prone to be inaccuracte._ 

**Idea**:
* decrease the size of key vector,
* increase the size of value vector and perform the addition of head outputs.

### Simple python setup
* I have tried on Python 3.9.2, 
(since the dependencies are kept as low as possible should be easy to run/adapt on other Python version.)
* PyTorch version 1.8.1
* Torchtext version 0.9.1
* Pandas version 0.9.1

### How to run the classifier?
```console
declare@lab:~$ python text_classifier.py -dataset data.csv
```
***Note***: Feel free to replace _data.csv_ with your choice of text classification problem, be it sentiment, news topic, reviews, etc.

#### data.csv (format)
should be two columns, header of the column with labels is "label" and text is "text". For example:

| text | label |
|-------|-----|
| we love NLP  | 5 |
| I ate too much, feeling sick   | 1  |

### In house datasets
BTW, you can try to run on Torchtext provided [datasets](https://pytorch.org/text/stable/datasets.html#id5) for classification. For AG_NEWS dataset,  
```console
declare@lab:~$ python text_classifier.py -kdim 16 -concat False -dataset ag_news
```
can replace _ag_news_ with _imdb_ for IMDb, _sogou_ for SogouNews, _yelp_p_ for YelpReviewPolarity
, _yelp_f_ for YelpReviewFull, _amazon_p_ for AmazonReviewPolarity, _amazon_f_ for AmazonReviewFull, _yahoo_ for YahooAnswers, _dbpedia_ for DBpedia.

#### Want to customize it for more identifiability?
Keep low k-dim and/or switch head addition by concat = False. Feel free to analyze attention weights for inputs with lengths up to embedding dim.

```console
declare@lab:~$ python text_classifier.py -kdim 16 -concat False -dataset ag_news
```
***Note***: Lower k-dim may/may not impact the classification accuracy, please keep the possible trade-off in the bucket during experiments.

#### Tweak classifier parameters
* batch: training batch size (default = 64).
* nhead: number of attention heads (default = 4).
* concat: (True: concatenate / False: addition) of attention head output (default = True).
* epochs: number training epochs (default = 10).
* lr: learning rate (default = 0.001).
* vocab_size: set threshold on vocabular size (default = 100000).
* max_text_len: trim the text longer than this value (default = 512).
* test_frac: only for user specified datasets, fraction of test set from the specified data set (default = 0.3).
* valid_frac: fraction of training samples kept aside for model development (default = 0.3).
* kdim: dimensions of key (and query) vector (default = 16).
* embedim: decides dimention of token vectors and value vector, i.e.,

| concat | vdim |
|-------|-----|
| True  | <img src="https://latex.codecogs.com/svg.latex?\small&space;\frac{\text{embedim}}{\text{nhead}}" title="\small \frac{\text{embedim}}{\text{nhead}}" />|
| False  | embedim |

