import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import math
import collections
import pickle
import time
import threading,queue


def remove_stop_words(all_tokens,stop_words):
    lemmatizer = WordNetLemmatizer()
    tokens_without_stop_words = [x for x in all_tokens if x not in stop_words]
    tokens_after_preprocess = []
    for word in tokens_without_stop_words:
        tokens_after_preprocess.append(lemmatizer.lemmatize(word))
    return tokens_after_preprocess

def compute_tf(tokens):
    term_freq = {}
    for word in tokens:
        if word in term_freq:
            term_freq[word] += 1
        else:
            term_freq[word] = 1
    for word in term_freq:
        term_freq[word] = term_freq[word] / len(tokens)
    return term_freq

def compute_df(words, doc_freq, title):
    for word in words:
        if word in doc_freq:
            doc_freq[word].append(title)
        else:
            temp = []
            temp.append(title)
            doc_freq[word] = temp
    return doc_freq

def compute_tf_idf(items, df):
    tf_idf = {}
    idf = {}
    for title in items:
        tf_idf[title] = {}
        for word in items[title]:
            idf[word] = math.log(len(items) / int(len(df[word])))
            tfidf = items[title][word] * idf[word]
            tf_idf[title][word] = tfidf
    return tf_idf, idf


def compute_idf(items, df):
    idf = {}
    for title in items:
        for word in items[title]:
            idf[word] = math.log(len(items) / int(df[word]))
    return idf


def compute_tfidfvector(descwords, wordDict):
    tfidfVector = [0.0] * len(wordDict)
    for i, word in enumerate(wordDict):
        if word in descwords:
            tfidfVector[i] = descwords[word]
    return tfidfVector

def dot_product(doc_vector, query_vector):
    dot = 0.0
    for e_x, e_y in zip(doc_vector, query_vector):
        dot += e_x * e_y
    return dot

def magnitude(vector):
    mag = 0.0
    for index in vector:
        mag += math.pow(index, 2)
    return math.sqrt(mag)

def cosine_similarity(doc_vector, q_vector):
    qdoc_similarity = dot_product(doc_vector, q_vector) / magnitude(doc_vector) * magnitude(q_vector)

def score_value(e_title,sim_score):
    for e_title in sim_score:
        if sim_score[e_title] > 0:
            match_document = e_title + " - " + str(sim_score[e_title])
    return match_document

def get_similarity(tf_idf, wordDict, qtfidfvector, title_list):
    temp = {}
    for e_title in title_list:
        qdoc_similarity = 0.0
        doc_tf_idf_vector = compute_tfidfvector(tf_idf[e_title], wordDict)
        magnitude_value = magnitude(doc_tf_idf_vector) * magnitude(qtfidfvector[0])
        if magnitude_value != 0.0:
            qdoc_similarity = dot_product(doc_tf_idf_vector, qtfidfvector[0]) / magnitude_value
        if qdoc_similarity != 0.0:
            temp[e_title] = qdoc_similarity
    return temp

def sort_dictionary(wordDict):
    sortedList = []
    for entry in sorted(wordDict.items(), reverse=True, key = lambda x: x[1]):
        sortedList.append(entry)
    return sortedList

def store_tfidfvector():
    Location = "/home/mbbhavana/BookSearch/static/data/"
    json_data = open(Location+"goodreads_books_young_adult.json", 'r')
    tf_idf_json = open(Location+"goodreads_books_young_adult_tf_idf.json", 'w')
    idf_json = open(Location+"goodreads_books_young_adult_idf.json", 'wb')
    word_list = open(Location+"goodreads_books_young_adult_word_list.json",'w')
    # with open("/home/mbbhavana/BookSearch/static/data/goodreads_books_young_adult.json",'rb') as json_data:
    title_desc = open(Location+"title_desc.json", 'wb')
    data_items = {}
    doc_freq = {}
    tf_idf = {}
    idf = {}
    i = 0
    for line in json_data:
        if i < 1000:
            each_book = json.loads(line)
            if each_book["title"] not in data_items:
                temp = {}
                temp["description"] = each_book["description"]
                data_items[each_book["title"]] = temp
                i += 1
            else:
                break
    pickle.dump(data_items, title_desc)
    title_desc.close()
    json_data.close()
    stop_words = set(stopwords.words('english'))
    for e_title in data_items:
        title_tokens = word_tokenize(e_title.lower())
        description_tokens = word_tokenize(data_items[e_title]["description"].lower())
        all_tokens = title_tokens + description_tokens
        final_tokens = remove_stop_words(all_tokens,stop_words)
        data_items[e_title] = compute_tf(final_tokens)
        doc_freq = compute_df(data_items[e_title], doc_freq, e_title)

    tf_idf, idf = compute_tf_idf(data_items, doc_freq)
    pickle.dump(idf, idf_json)
    idf_json.close()
    wordDict = sorted(doc_freq.keys())
    json.dump(doc_freq,word_list)
    word_list.close()
    #for e_title in tf_idf:
    #    tf_idf_vector = compute_tfidfvector(tf_idf[e_title], wordDict)
    #    temp = {}
    #    temp[e_title] = tf_idf_vector
    #    pickle.dump(temp,tf_idf_json)
    json.dump(tf_idf,tf_idf_json)
    tf_idf_json.close()

def return_search(query,stop_words):
    Location = "/home/mbbhavana/BookSearch/static/data/"
    search_tf = {}
    querytokens = word_tokenize(query.lower())
    query_tokens = remove_stop_words(querytokens,stop_words)
    idf_json = open(Location+"goodreads_books_young_adult_idf.json", 'rb')
    word_list = open(Location+"goodreads_books_young_adult_word_list.json",'r')
    desc_list = open(Location+"title_desc.json", 'rb')
    tf_idf_json = open(Location+"goodreads_books_young_adult_tf_idf.json", 'r')
    title_desc = pickle.load(desc_list)
    desc_list.close()
    idf = pickle.load(idf_json)
    idf_json.close()
    doc_freq = json.load(word_list)
    word_list.close()
    wordDict = sorted(doc_freq.keys())
    title_list = []
    for i in query_tokens:
        if i not in wordDict:
            query_tokens.remove(i)
        else:
            temp = doc_freq[i]
            for title in temp:
                title_list.append(title)
    if len(query_tokens) == 0:
        return 0
    search_tf = compute_tf(query_tokens)
    for key in search_tf:
        if key in idf:
            search_tf[key] = search_tf[key] * idf[key]
        else:
            search_tf[key] = 0.00
    qtfidfvector = [compute_tfidfvector(search_tf, wordDict)]
    sim_score = {}
    tf_idf = {}
    tf_idf = json.load(tf_idf_json)
    tf_idf_json.close()
    start = time.time()
    sim_score = get_similarity(tf_idf, wordDict, qtfidfvector, title_list)
    print("Comparision time: %s seconds" %(time.time() - start))
    docslist = sort_dictionary(sim_score)
    list_len = len(docslist)
    i = 0
    listResult = []
    simscore = []
    if list_len != 0:
        while i < len(docslist):
            temp = {}
            simscore = docslist[i][1]
            desc = title_desc[docslist[i][0]]["description"]
            titleList = docslist[i][0]+"\n(Similarity score:"+str(simscore)+")"
            temp[titleList] = str(desc)
            listResult.append(temp)
            i += 1
    else:
        return(0)
    return(listResult,query_tokens)

if __name__ == '__main__':
    store_tfidfvector()
