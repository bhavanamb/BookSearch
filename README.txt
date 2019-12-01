# vim: set ft=rst:

See https://help.pythonanywhere.com/ (or click the "Help" link at the top
right) for help on how to use PythonAnywhere, including tips on copying and
pasting from consoles, and writing your own web applications.

Project  Name : Book Search

Technology Used: Python 3.7, flask

Installation:
Deploy this project in any flask based web hosting service
Step1:Create virtual environment with python 3.7
step2:Install the following packages needed for this project
    nltk(pip install nltk)
    logon to python console: Run following commands
    Import nltk
    nltk.download(punkt)
    nltk.download(stopwords)
    nltk.download('wordnet')
Step3: pip install flask

In search.py
1)Change location of datafiles based on the project location
2)This file consists of tf-idf algorithm code.
