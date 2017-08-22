# First mandatory assignment: baby's first machine learning

In this assignment you will parallellize the training of a classifier. I provide two datasets: 
* `spam.csv` [from UCI ML](https://archive.ics.uci.edu/ml/datasets/spambase) describes a collection of emails marked as spam or not (labelled 1 or 0). You want to predict whether an email is spam or not. The predictors (or features) in these data are things such as how often a certain word appears &c. This is a smallish dataset.
* `creditfraud.csv` [from kaggle](https://www.kaggle.com/dalpozz/creditcardfraud) describes credit card transactions, some of which are fraudulent (labelled 1 or 0). You want to detect fraudulent transactions. The features are nondescript for privacy reasons. This dataset is considerably larger.

I have normalized both these data sets, which is why you won't recognize the word count frequencies in the spam set as frequencies. 

The sequential program in `precode.py` fits a logistic regression model by batch gradient descent. Gradient descent is a numerical method for finding an optimum in a function with well-defined first derivative. It works on subsets of data, or batches, to iterate toward the best solution. [Andrew Ng has some excellent lecture notes on this for those of you who are interested](http://cs229.stanford.edu/notes/cs229-notes1.pdf). Andrew writes about stochastic GD, which is the same as batch GD with a single sample per batch.

# What I expect from you
* Parallellize the sequential program. You can do this in Python, but beware that (i) while the standard python thread package gives you the primitives of threading, it actually runs in sequence. For True but slow parallellism, use `multiprocessing`. And (ii) python is kind of slow for a lot of things. 
* Write a little report to go with it and **report some numbers**. Is the parallel implementation faster? Why? Or if it isn't: why not?

It's fine to do Python and just modify the precode. If you want to you can use any language within reason if you're willing to roll you own precode. I would recommend Go with goroutines, which is probably perfect for this exercise. C and ptreads also works.

# Handin
The usual:
* Code in `src/`
* Report in `doc/`

# If you finish early and want to do a fun and easy-to-parallelize ML algorithm:
Implement random forests (in parallel). 

# A very good resource if you're interested in prediction
[The excellent and free Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) describes logistic regression, random forests, and more.

# Disclaimer
This example is a bit forced and probably doesn't lead a very good solution. Partly because of how I have programmed this and how I set the different parameters, and partly because gradient descent like this isn't the usual way to fit a logistic regression.
