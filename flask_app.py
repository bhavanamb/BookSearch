
# A very simple Flask Hello World app for you to get started with...

from datetime import datetime
from flask import Flask,render_template,request
import search
import time
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/search', methods=['GET', 'POST'])
def search_results():
    if request.method == "POST":
        searchkey = request.form['search']
        start = time.time()
        similar_title,highlight_tokens = search.return_search(searchkey,stop_words)
        highlight = " ".join(highlight_tokens)
        print("Comparision Time:")
        print("%s seconds" %(time.time() - start))
        if similar_title == 0:
                return("No Results found for query: %s" %(searchkey))
        else:
            display_titledesc = {}
            for i in similar_title:
                title = list(i.keys())[0]
                display_titledesc[title]= i[title]
            #print(display_titledesc)
            return render_template("result.html", document = display_titledesc, skey = highlight)

