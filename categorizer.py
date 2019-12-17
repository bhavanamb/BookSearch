import json
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import itertools

def get_tokens(tokenizer,document):
    tokens = tokenizer.tokenize(document.lower())
    return tokens

def remove_stop_words(stop_words, all_tokens):
    portstem = PorterStemmer()
    tokens_without_stop_words = [x for x in all_tokens if x not in stop_words]
    tokens_after_preprocess = []
    for word in tokens_without_stop_words:
        tokens_after_preprocess.append(portstem.stem(word))
    return tokens_after_preprocess

def sort_dictionary(wordDict):
    sortedList = []
    for entry in sorted(wordDict.items(), reverse=True, key = lambda x: x[1]):
        sortedList.append(entry)
    return sortedList

def get_tokens_word_list(token_dic,word_list,book_word_count):
    json_data = open("C:\\Users\\tejes\\Downloads\\bhavana\\Data Mining\\Project\\Datasets\\classify_test.json", 'r')
    genre_list = ["fiction", "science-fiction", "mystery", "horror", "fantasy", "romantic"]
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    token_dic = {}
    word_list = {}
    book_word_count = {}
    i = 0
    for line in json_data:
        i += 1
        each_book = json.loads(line)
        title_tokens = get_tokens(tokenizer,each_book['title'])
        desc_tokens = get_tokens(tokenizer,each_book['description'])
        all_tokens = title_tokens+desc_tokens
        final_tokens = remove_stop_words(stop_words, all_tokens)
        book_id = each_book['book_id']
        temp = {}
        temp_wc = {}
        for each_word in final_tokens:
            if each_word not in word_list:
                word_list[each_word] = 1
            else:
                word_list[each_word] = word_list[each_word] + 1
            if each_word not in temp_wc:
                temp_wc[each_word] = 1
            else:
                temp_wc[each_word] = temp_wc[each_word] + 1
            book_word_count[book_id] = temp_wc
            temp[each_word] = each_book['popular_shelves']
        token_dic[each_book['book_id']] = temp
    return token_dic,book_word_count,word_list

#genre as index : word and count in nested dic
def get_genre_list(token_dic,book_word_count,store_list):
    for each_book in token_dic:
        for each_word in token_dic[each_book]:
            templist = []
            templist = token_dic[each_book][each_word]
            for each_item in templist:
                genre = each_item['name']
                if genre in store_list:
                    if each_word in store_list[genre]:
                        store_list[genre][each_word] = store_list[genre][each_word] + book_word_count[each_book][each_word]
                    else:
                        store_list[genre][each_word] = book_word_count[each_book][each_word]
                else:
                    temp = {}
                    temp[each_word] = book_word_count[each_book][each_word]
                    store_list[genre] = temp
    return store_list

def get_total_word_count(store_list):
    count_eachword = {}
    for genre in store_list:
        for each_word in store_list[genre]:
            if each_word in count_eachword:
                count_eachword[each_word] += store_list[genre][each_word]
            else:
                count_eachword[each_word] = store_list[genre][each_word]
    return count_eachword

def calc_post_prob(store_list,count_eachword,word_list):
    genre_wordcount = {}
    for genre in store_list:
        genre_wordcount[genre] = sum(store_list[genre].values())
        #print(genre_wordcount)
    total_genre_wordcount = sum(genre_wordcount.values())
    #print(total_genre_wordcount)
    p_genre = {}
    #Calculating posterior probabilities
    for genre in store_list:
        for each_word in store_list[genre]:
            # laplace smoothing
            prob_wordgiven_genre = ((store_list[genre][each_word])+0.01)/(genre_wordcount[genre])+len(word_list)
            prob_wordgiven_notgenre = ((count_eachword[each_word] - store_list[genre][each_word])+0.01/(total_genre_wordcount - genre_wordcount[genre])+len(word_list)
            temp = []
            temp.append(prob_wordgiven_genre)
            temp.append(prob_wordgiven_notgenre)
            store_list[genre][each_word] = temp
        #calculating prior probabilities
        temp_gen = []
        genre_p = genre_wordcount[genre] / total_genre_wordcount
        temp_gen.append(genre_p)
        genre_np = (total_genre_wordcount - genre_wordcount[genre])/total_genre_wordcount
        temp_gen.append(genre_np)
        p_genre[genre] = temp_gen
    return store_list, p_genre

def calc_search_prob(tokens, store_list, pgenre):
    p_gen_word = {}
    len_tokens = len(tokens)
    display_prob_list = {}
    temp = {}
    j = 0
    for genre in store_list:
        #print(each_token)
        tokens_prod = 1
        token_notgenre = 1
        for each_token in tokens:
            list_displ = []
            if each_token in store_list[genre]:
                while j<1:
                    list_displ = [store_list[genre][each_token][0],store_list[genre][each_token][1],tokens_prod*pgenre[genre][0],token_notgenre*pgenre[genre][1]]
                    temp[each_token] = list_displ
                    display_prob_list[genre] = temp
                    j=j+1
                tokens_prod = tokens_prod * store_list[genre][each_token][0]
                token_notgenre = token_notgenre * store_list[genre][each_token][1]
                denom = (token_notgenre*pgenre[genre][1]) + (tokens_prod*pgenre[genre][0])
                p_gen_word[genre] = ((tokens_prod*pgenre[genre][0])/denom)*100
    return p_gen_word,display_prob_list

def search_probability(search_text):
    Location = "/home/mbbhavana/BookSearch/static/data/"
    prior_prob = open(Location+"prior_prob.json", 'r')
    posterior_prob = open(Location+"posterior_prob.json", 'r')
    store_list = json.load(posterior_prob)
    p_genre = json.load(prior_prob)
    search_tokens = []
    p_list = []
    temp_word_p = {}
    tokenizer = RegexpTokenizer(r'\w+')
    temp_search_tokens = get_tokens(tokenizer,search_text)
    stop_words = set(stopwords.words('english'))
    search_tokens = remove_stop_words(stop_words, temp_search_tokens)
    p_genreword,display_list = calc_search_prob(search_tokens,store_list, p_genre)
    p_list = sort_dictionary(p_genreword)
    list_len = len(p_list)
    i = 0
    temp = {}
    if list_len != 0:
        while i < len(p_list):
            prob_gen = p_list[i][1]
            gen = p_list[i][0]
            temp[gen] = str(prob_gen)
            i += 1
    #print(temp)
    n = 1
    #display_score = dict(itertools.islice(display_list.items(), n))
    posterior_prob.close()
    prior_prob.close()
    print(display_list)
    return temp,display_list

if __name__ == "__main__":
    #posterior_prob = open("C:\\Users\\tejes\Downloads\\bhavana\\Data Mining\\Project\\classifier_files\\posterior_prob.json",'w')
    #prior_prob = open("C:\\Users\\tejes\Downloads\\bhavana\\Data Mining\\Project\\classifier_files\\prior_prob.json",'w')
    token_dic = {}
    book_word_count = {}
    word_list = {}
    count_eachword = {}
    p_genre = {}
    store_list = {}
    print("Get_Tokens")
    #token_dic,book_word_count,word_list = get_tokens_word_list(token_dic,book_word_count,word_list)
    print("Get Store List")
    #store_list = get_genre_list(token_dic,book_word_count,store_list)
    print("Get Word Count")
    #count_eachword = get_total_word_count(store_list)
    #store_list,p_genre = calc_post_prob(store_list,count_eachword,word_list)
    #json.dump(store_list, posterior_prob)
    #json.dump(p_genre, prior_prob)
    #posterior_prob.close()
    #prior_prob.close()
    #search_text = input("enter query!")
    #search_probability(search_text)
