from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import json
import math
import time

def get_tokens(tokenizer,document):
    tokens = tokenizer.tokenize(document.lower())
    return tokens

def remove_stop_words(all_tokens,stop_words):
    lemmatizer = WordNetLemmatizer()
    portstem = PorterStemmer()
    snowstem = SnowballStemmer("english")
    tokens_without_stop_words = [x for x in all_tokens if x not in stop_words]
    tokens_after_preprocess = []
    for word in tokens_without_stop_words:
        tokens_after_preprocess.append(lemmatizer.lemmatize(word))
        #tokens_after_preprocess.append(portstem.stem(word))
        #tokens_after_preprocess.append(snowstem.stem(word))
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
    #print(doc_freq)
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

def image_search_compute():
    Location = "/home/mbbhavana/BookSearch/static/data"
    captions_dataset = open(Location+"img_captions_set1.json", 'r')
    captn_tf_idf_json = open(Location + "captions_tf_idf.json", 'w')
    captn_idf_json = open(Location + "captions_idf_idf.json", 'w')
    doc_freq_json = open(Location + "doc_freq_idf.json", 'w')
    store_captions=json.load(captions_dataset)
    i = 0
    token_dic = {}
    tf_dict = {}
    doc_freq = {}
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    for each_item in store_captions:
        i+=1
        try:
            cap_text = store_captions[each_item][1]
        except:
            pass
        tempc_tokens = get_tokens(tokenizer, cap_text)
        capt_tokens =   remove_stop_words(tempc_tokens, stop_words)
        token_dic[each_item] = capt_tokens
        tf_dict[each_item] = compute_tf(capt_tokens)
        doc_freq = compute_df(tf_dict[each_item],doc_freq,each_item)

    tf_idf_captn,idf_captn = compute_tf_idf(tf_dict,doc_freq)
    json.dump(idf_captn,captn_idf_json)
    captn_idf_json.close()
    json.dump(doc_freq,doc_freq_json)
    doc_freq_json.close()
    json.dump(tf_idf_captn,captn_tf_idf_json)
    captn_tf_idf_json.close()

def caption_search(query):
    start_time = time.time()
    stop_words = set(stopwords.words('english'))
    Location = "/home/mbbhavana/BookSearch/static/data/"
    tokenizer = RegexpTokenizer(r'\w+')
    tempq_tokens = get_tokens(tokenizer, query)
    query_tokens = remove_stop_words(tempq_tokens, stop_words)
    captn_idf_json = open(Location + "captions_idf_idf.json", 'r')
    doc_freq_json = open(Location + "doc_freq_idf.json", 'r')
    captn_tf_idf_json = open(Location + "captions_tf_idf.json", 'r')
    captions_dataset = open(Location + "final_captions.json", 'r')
    title_caption = json.load(captions_dataset)
    captn_idf = json.load(captn_idf_json)
    word_doc_list =json.load(doc_freq_json)
    captn_tf_idf = json.load(captn_tf_idf_json)
    word_dic = sorted(word_doc_list.keys())
    print("Time taken to load data:%s"%(time.time()-start_time))
    #print(word_dic)
    title_list = []
    ind_list = []
    phr_list = []
    display_list = {}
    for i in query_tokens:
        if i not in word_dic:
            query_tokens.remove(i)
        else:
            temp = word_doc_list[i]
            print("PHR:%s"%(phr_list))
            if len(phr_list) == 0:
                for title in temp:
                    ind_list.append(title)
                    phr_list.append(title)
            else:
                phr_list = list(set(phr_list).intersection(set(temp)))
                for title in temp:
                    if title not in ind_list:
                        ind_list.append(title)

    if len(phr_list) != 0:
        title_list = phr_list
    else:
        title_list = ind_list
    if len(query_tokens) == 0:
        return(0)
    search_tf = compute_tf(query_tokens)
    for key in search_tf:
        if key in captn_idf:
            search_tf[key] = search_tf[key] * captn_idf[key]
        else:
            search_tf[key] = 0.00
    qtfidfvector = [compute_tfidfvector(search_tf, word_dic)]
    sim_score = get_similarity(captn_tf_idf, word_dic, qtfidfvector, title_list)
    docslist = sort_dictionary(sim_score)
    print(len(docslist))
    captn_idf_json.close()
    doc_freq_json.close()
    captn_tf_idf_json.close()
    list_len = len(docslist)
    i = 0
    disply_dic = {}
    if list_len != 0:
        while i < len(docslist):
            temp = {}
            display_lst = []
            simscore = docslist[i][1]
            title = docslist[i][0]
            list_details = title_caption[title]
            #print(list_details)
            try:
                cap_text = list_details[1]
                img_url = list_details[0]
            except:
                print(list_details)
            display_lst.append(cap_text)
            display_lst.append(img_url)
            display_lst.append(simscore)
            disply_dic[title] = display_lst
            i += 1
    else:
        return(0)
    #print(disply_dic)
    print("Time tken:%s"%(time.time()-start_time))
    return disply_dic,query_tokens

if __name__ == "__main__":
    image_search_compute()
    #print("Success")
    #query = input("enter text:")
    #caption_search(query)
