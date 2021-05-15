import re
from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

def get_top_n(dict_elem, n, frequency):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n])
    for key in result.keys():
        result[key] = frequency[key]
    return result




def get_keywords(doc):
    doc = doc.lower()
    stop_words = set(stopwords.words('english'))

    doc = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', doc)
    doc = re.sub(r'[^\w\s\?]', ' ', doc)
    # Remove some special characters
    doc = re.sub(r'([\;\:\|•«\n])', ' ', doc)

    total_words = doc.split()
    total_word_length = len(total_words)
    print(total_word_length)
    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)
    print(total_sent_len)
    frequency = {}
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
                frequency[each_word] += 1
            else:
                tf_score[each_word] = 1
                frequency[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    print("tf score: ", tf_score)

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

    print("idf-score: ", idf_score)
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    print("tf-idf score: ", tf_idf_score)
#    print("Top Keywords: ", get_top_n(tf_idf_score, 30, frequency))
    return get_top_n(tf_idf_score, 30, frequency)
    
    

    
