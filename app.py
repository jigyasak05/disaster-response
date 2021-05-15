from flask import Flask, render_template, request, url_for
import pandas as pd
import torch
import os
import re
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizerFast
import snscrape.modules.twitter as sntwitter
from flask import jsonify
from flask_cors import CORS, cross_origin
from keywords_extraction import get_keywords
import heapq

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
#tweets_file = "data/fetched_tweets.csv"
@app.route("/")
@app.route("/home")
def home():
 return render_template("index.html")

#@app.route('/')
#def root():
#	return render_template('test.html')

# @app.route('/search',methods=['POST'])
# def search():
# 	if request.method == 'POST':
#    		keyword = request.json
#    		print(keyword)
# 	return send_response()

@app.route('/search',methods=['POST'])
def search():
    #loaded_model = pickle.load(open('model.sav', 'rb'))
    if request.method == 'POST':
#        keyword = request.form['query']
        keyword = request.json['query']
        fetch_tweets(keyword)
        all_tweets = save_preprocessed()
        top_keywords = get_keywords(all_tweets)
        prediction_dataloader = tokenize_tweets()
        top_tweets, label_count = predict(prediction_dataloader)
        prediction = len(top_tweets)
        for list in top_tweets:
            print(list)
        return make_json(label_count, top_keywords, top_tweets)
#       return render_template("test.html",prediction=prediction)

def fetch_tweets(keyword):

    tweets_list = []
    # TODO: change date and call for both languages
    query_string = "{} since:2021-02-06 lang:en".format(keyword)

    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query_string).get_items()):    
        if i>100:
            break

        # TODO: Remove retweets here if possible, break loop on the basis of count
        # TODO: Cleaning if possible (RE not in tweets)
        tweets_list.append([tweet.date, str(tweet.id), tweet.content])

    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet ID', 'Tweet text'])

    tweets_df.to_csv("data/fetched_tweets.csv")

def preprocess_tweets(tweet):
    
    tweet_urls = re.findall('https?://t\.co/\S+', tweet)
    tweet = re.sub(r'https?://t\.co/\S+', '', tweet)

    # Remove username
    tweet = re.sub('@[^\s]+', '', tweet)

    # re.sub(r'(@.*?)[\s]', ' ', "@dsbvd xyz @ebhjw dhejwbdj @nedjk")
    # re.sub('@[^\s]+','',"@dsbvd xyz @ebhjw dhejwbdj @nedjk")

    # Replace '&amp;' with '&', '&lt;' with '<', '&gt;' with '>'
    tweet = re.sub(r'&amp;', '&', tweet)
    tweet = re.sub(r'&gt;', '>', tweet)
    tweet = re.sub(r'&lt;', '<', tweet)

    # Remove trailing whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    # Remove '#' symbol
    tweet = re.sub(r'#', '', tweet)
    
    return tweet

def save_preprocessed():
    all_tweets = ""
    tweets_processed = pd.DataFrame(columns = ['tweet_id', 'tweet_text'])  # new empty dataframe
    tweets = pd.read_csv('data/fetched_tweets.csv')
    tweets = tweets[['Tweet ID','Tweet text']]

    for i in range(0, tweets.shape[0]):
      
      tweet = tweets.loc[i,'Tweet text']
      processed_tweet = preprocess_tweets(tweet)
      all_tweets += processed_tweet
      tweets_processed = tweets_processed.append({'tweet_id': tweets.loc[i,'Tweet ID'], 'tweet_text': processed_tweet}, ignore_index=True)

    tweets_processed.to_csv('data/final_processed_dataset.csv')
    return all_tweets

def tokenize_tweets():

    tweets_processed = pd.read_csv('data/final_processed_dataset.csv')
    sentences = tweets_processed.tweet_text.values

    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Set the batch size.
    batch_size = 32 

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print("Tokenization has completed!")
    return prediction_dataloader

class pair:
    prob = 0.0
    idx = -1

    def __init__(self, probability, index):
        self.prob = probability
        self.idx = index

    def __lt__(self, other):
        return self.prob >= other.prob

def predict(prediction_dataloader):
    # Put model in evaluation mode
    model_loaded.eval()

    tweets = pd.read_csv('data/fetched_tweets.csv')
    tweets = tweets[['Tweet ID','Tweet text']]

    # predictions = []
    batch_list = []

    # Predict 
    for batch in prediction_dataloader:
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
      
      # Telling the model not to compute or store gradients, saving memory and 
      # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model_loaded(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        # # torch.topk(logits, 5, 1)
        # idx = logits.argmax()
        # max_lbl_prob = logits[idx]


        probability = torch.nn.functional.sigmoid(logits)
        batch_list.append(probability.tolist())
        # Store predictions and true labels
        # predictions.append(logits)

    index_list = [[]]
    for itr in range(6):
        sublist = []
        index_list.append(sublist)

    print("batch list has formed!")

    count = 0
    label_count = [0] * 6
    
    for i in range(len(batch_list)):
        for j in range(len(batch_list[i])):

            #probability_list.append(batch_list[i][j])
            max_value = max(batch_list[i][j])
            max_index = batch_list[i][j].index(max_value)
            label_count[max_index] += 1
            p = pair(max_value, count)
            index_list[max_index].append(p)

            count = count + 1
            #Add to respective heap

    print("probability_list has formed!")

    return get_top_tweets(index_list), label_count


def get_top_tweets(sorted_list):

    tweets = pd.read_csv('data/fetched_tweets.csv')
    tweets = tweets[['Tweet text']]

    max_tweets_list = [[]]
    for itr in range(6):
        sublist = []
        max_tweets_list.append(sublist)

    for i in range(6):
        sorted_list[i].sort()
        sorted_list[i] = sorted_list[i][:10]

        for j in range(len(sorted_list[i])):
            p = sorted_list[i][j]
            index = p.idx
            tweet = tweets.loc[index,'Tweet text']
            max_tweets_list[i].append(tweet)

    return max_tweets_list


# temporary function
def send_response():

	data = {"keywords": ["k1", "k2", "k3", "k4", "k2", "k1", "k1", "k2"]}

	return jsonify(data)

def make_json(label_count, keywords, top_tweets):
    label_mapping = {
        0: "affected_individuals",
        1: "caution_and_advice",
        2: "donation_and_volunteering",
        3: "infrastructure_and_utility_damage",
        4: "sympathy_and_moral_support",
        5: "not_relevant"
    }
    label_count_dict = {}
    total_count = 0
    for i in range(6):
        total_count = total_count + label_count[i]
        key = label_mapping[i]
        label_count_dict[key] = label_count[i]
    label_count_dict["total"] = total_count
    top_tweets_dict = {}
    for i in range(6):
        top_tweets_dict[label_mapping[i]] = top_tweets[i]
    print("DICTIONARY OF LABEL COUNT: ", label_count_dict)
#    data = {"en": {"keywords": jsonify(keywords), "count": jsonify(label_count_dict), "top_tweets": top_tweets}, "hi": {"keywords": jsonify(keywords), "count": jsonify(label_count_dict), "top_tweets": top_tweets}}
    data = {"en": {"keywords": keywords, "count": label_count_dict, "top_tweets": top_tweets_dict}, "hi": {"keywords": keywords, "count": label_count_dict, "top_tweets": top_tweets_dict}}
    
    return data
# jsonify andar wali dictionary -TODO maybe?

if __name__ == "__main__":
    output_dir = "data/xlm-roberta_model_save"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(output_dir)
    model_loaded = XLMRobertaForSequenceClassification.from_pretrained(output_dir)
    app.run(debug=True)

