# Book Search #

##Technology Used: Python 3.7, flask##

##Installation:##
###Deploy this project in any flask based web hosting service###
*Step1:Create virtual environment with python 3.7
*Step2:Install the following packages needed for this project
    *nltk(pip install nltk)
    *logon to python console: Run following commands
    *Import nltk
    *nltk.download(punkt)
    *nltk.download(stopwords)
    *nltk.download('wordnet')
*Step3: pip install flask
*Step4: pip install Flask-Cache  
*Step5: Change location of datafile in search.py,categorizer.py and image_search.py based on project datafile location
In this project datafiles location:"/home/mbbhavana/BookSearch/static/data/"

##There are 3 phases in this project##

1. Text Search
In search.py
This file consists of tf-idf algorithm code.
In this matching documents are retrieved based on similarity score between query and document.

2.Classifier
Multilabel Naive Bayes Classifier
Classification is identifying to which class new data belongs to, given a number of classes.
In machine learning classifier utilizes training data to predict the class of data input given to it. In the Book data set, each book can have one or more classes, this is a multi label classification.

3.Image Search
Generated captions for images and implemented the search using tf-idf algorithm.
