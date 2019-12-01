
# A very simple Flask Hello World app for you to get started with...

from datetime import datetime
from flask import Flask,render_template,request,flash
import search
import categorizer
import image_search
import time
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.secret_key = 'SECRET KEY'

@app.route('/')
def index():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/imgsearch')
def imgsearch():
    """Renders the contact page."""
    return render_template(
        'image_index.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'classify.html',
        title='Classify',
        year=datetime.now().year,
    )

@app.route('/search', methods=['GET', 'POST'])
def search_results():
    if request.method == "POST":
        searchkey = request.form['search']
        start = time.time()
        similar_title,highlight_tokens,sim_score,tfidf_display,idf_dict = search.return_search(searchkey,stop_words)
        print("Comparision Time:")
        print("%s seconds" %(time.time() - start))
        if highlight_tokens == 0 and similar_title == 0 and sim_score==0 and tfidf_display==0:
            flash("NO RESULTS FOUND")
            return render_template("index.html")
        else:
            highlight = " ".join(highlight_tokens)
            display_titledesc = {}
            for i in similar_title:
                title = list(i.keys())[0]
                display_titledesc[title]= i[title]
            #print(display_titledesc)
            return render_template("results.html", document = display_titledesc, skey = highlight, sim = sim_score, tfidf = tfidf_display, idfvalue = idf_dict)

@app.route('/classify', methods = ['GET', 'POST'])
def classify_results():
    if request.method == "POST":
        classifykey = request.form['classify']
        classify_prob,word_prb = categorizer.search_probability(classifykey)
    return render_template("classifyresult.html", document = classify_prob, prob = word_prb)

@app.route('/image', methods = ['GET', 'POST'])
def image_results():
    if request.method == "POST":
        imagekey = request.form['image']
        results, highlight_tokens = image_search.caption_search(imagekey)
        if results == 0 and highlight_tokens == 0:
            flash("NO RESULTS FOUND")
            return render_template("image_index.html")
        else:
            highlight = " ".join(highlight_tokens)
            return render_template("imageresult.html", document = results, skey = highlight)
