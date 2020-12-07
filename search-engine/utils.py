from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from collections import Counter, defaultdict
from chronometer import Chronometer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from numpy import linalg as la
from array import array
from config import *
import pandas as pd
import configparser
import numpy as np
import jsonlines
import datetime
import argparse
import twarc
import time
import json
import math
import glob
import csv
import re
import os

class bcolors:
    GRAY = '\033[37m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    DIM = '\033[2m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[35m'
    UNDERLINE = '\033[4m'

#######################
### DATA COLLECTION ###
#######################
class MyStreamListener(StreamListener):
    """
    Twitter listener, collects streaming tweets and output to a file
    """
    def __init__(self, api, OUTPUT_FILENAME, stop_condition=10):
        """
        initialize the stream, with num. of tweets and saving the outputfile
        """
        # this line is needed to import the characteristics of the streaming API
        super(MyStreamListener, self).__init__()
        # to-count the number of tweets collected
        self.num_tweets = 0
        # save filename
        self.filename = OUTPUT_FILENAME
        # stop-condition
        self.stop_condition = stop_condition

    def on_status(self, status):
        """
        this function runs each time a new bunch of tweets is retrived from the streaming
        """
        with open(self.filename, "a+") as f:
            tweet = status._json
            f.write(json.dumps(tweet) + '\n')
            #self.output.append(tweet)
            self.num_tweets += 1
            # Stop condition
            if self.num_tweets <= self.stop_condition:
                return True
            else:
                return False
        
    def on_error(self, status):
        """
        function useful to handle errors. It's possible to personalize it
        depending on the way we want to handle errors
        """
        print(status)
        #returning False in on_error disconnects the stream
        return False
        
# retrieve original tweets from their ids
def hydrate(ids, path, filename):
    """
    Hydrate tweets in order to update the data
    """
    T = twarc.Twarc()
    last_t = 0
    with Chronometer() as t:
        count = 0
        hydrated_tweets = []

        for tweet in T.hydrate(iter(ids)):
            assert tweet['id_str']
            count += 1
            hydrated_tweets.append(tweet)
            if(int(float(t)) % 10 == 0 and int(float(t)) != last_t):
                    print("Hydrated tweets:", len(hydrated_tweets))
                    last_t = int(float(t))
        
        with jsonlines.open(path + filename, mode='w') as writer:
            for obj in hydrated_tweets:
                writer.write(obj)
    
    return count, hydrated_tweets
        
def get_tweets(keywords, sample_size, mode = "read", data_directory = '../data/'):
    """
    Function that captures incoming tweets with the specified characteristics and
    returns the tweets as a list of jsons.
        if twitter_scraping = hydrate --> listen to incoming tweets and dump them into
                              json files in batches of 100
        if twitter_scraping = scrape --> listen to incoming tweets and dump them into
                              json files in batches of 100
        if twitter_scraping = read --> only read the already captured data stored in
                              json files
                              
    """
    
    # SCRAPE
    if (mode == "scrape"):
        # create 100 files of 1000 tweets each
        stop_conditions = [1000] * sample_size
        counter = 0
        for stop_condition in stop_conditions:
            counter += 1
            OUTPUT_FILENAME = data_directory + str(counter) + ".json"

            l = MyStreamListener(api, OUTPUT_FILENAME, stop_condition)
            # here we recall the Stream Class from Tweepy to input the authentication info and our personalized listener
            stream = Stream(auth=api.auth, listener=l)
            
            stream.filter(
                track=keywords,
                is_async=False,
                languages = ["en"]
            )
            
        # read all the files created in the data directory and
        # create a single data file named "final_data.json"
        filenames = glob.glob(data_directory + "*.json")
        data = []

        for filename in filenames:
            with jsonlines.open(filename) as reader:
                for obj in reader:
                    data.append(obj)
                
        with jsonlines.open(data_directory + "final_data.json", mode='w') as writer:
            for obj in data:
                writer.write(obj)
            
        # erase all single data files
        for filename in filenames:
            if filename != data_directory + "final_data.json":
                try: os.remove(filename)
                except: continue
            
    # HYDRATE
    elif (mode == "hydrate"):
        data = []
        with jsonlines.open(data_directory + "final_data.json") as reader:
            for obj in reader:
                data.append(obj)
        
        ids = [tweet["id_str"] for tweet in data]
    
        total_hydrated, data = hydrate(ids, data_directory, "final_data.json")
        # print(bcolors.GREEN + "Total tweets hydrated = " + str(total_hydrated) + bcolor.ENDC)

        
    # READ
    elif (mode == "read"):
        data = []
        with jsonlines.open(data_directory + "final_data.json") as reader:
            for obj in reader:
                data.append(obj)
            
    return data


########################
### DOCUMENT RANKING ###
########################
def normalize_text(text, tweet = False):
    '''
    Normalize text given a string
    '''
    # remove "RT" string indicating a retweet
    text = re.sub('RT', '', text)

    # lowering text
    text = text.lower()

    # removing all the punctuations
    if tweet: text = re.sub(r'[^\w\s#]', '', text).strip()
    else: text = re.sub(r'[^\w\s]', '', text).strip()

    # tokenize the text
    lst_text = text.split()
    
    # remove stopwords
    STOPWORDS = set(stopwords.words("english"))
    lst_text = [x for x in lst_text if (x not in STOPWORDS)]
    
    return lst_text

    
def get_normalized_tweet(tweet):
    '''
    Normalize tweets (taking into account hashtags and not urls) and return a list of terms
    '''
    # get text and entities of the tweet
    try: text = tweet["full_text"]
    except: text = tweet["text"]
    entities = tweet["entities"]
    
    lst_text = normalize_text(text, tweet = True)
    
    # remove hashtags and urls
    lst_urls = [re.sub(r'[^\w\s#]', '', urls["url"]).lower() for urls in entities["urls"]]
    lst_text = [x for x in lst_text if ("#" not in x) and (x not in lst_urls)]
    
    # add hashtags to lst_text (twice each hashtag -> double importance )
    for w in [hashtag["text"] for hashtag in entities["hashtags"]]:
        lst_text.append(normalize_text(w)[0])
        lst_text.append(normalize_text(w)[0])
    
    return lst_text

def create_index_tfidf(tweets, num_tweets):
    """
    Implement the inverted index and compute tf, df and idf
    
    Argument:
    lines -- collection of Wikipedia articles
    numDocuments -- total number of documents
    
    Returns:
    index - the inverted index (implemented through a python dictionary) containing terms as keys and the corresponding
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """
        
    index=defaultdict(list)
    tf=defaultdict(list) #term frequencies of terms in documents (documents in the same order as in the main index)
    df=defaultdict(int)         #document frequencies of terms in the corpus
    titleIndex=defaultdict(str)
    idf=defaultdict(float)
    
    for tweet in tweets:
        tweet_id = int(tweet["id_str"])
        terms = get_normalized_tweet(tweet) # normalized tweet text

        termdictPage={}

        for position, term in enumerate(terms): ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corrisponding list
                termdictPage[term][1].append(position)
            except:
                # add the new term as dict key and initialize the array of positions and add the position
                termdictPage[term]=[tweet_id, array('I',[position])] #'I' indicates unsigned int (int in python)
        
        # normalize term frequencies
        # compute the denominator to normalize term frequencies
        # norm is the same for all terms of a document.
        norm=0
        for term, posting in termdictPage.items():
            # posting is a list containing tweet_id and the list of positions for current term in current document:
            # posting ==> [current tweet, [list of positions]]
            # you can use it to inferr the frequency of current term.
            norm+=len(posting[1])**2
        norm=math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in termdictPage.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4))
            # increment the document frequency of current term (number of documents containing the current term)
            df[term]= df[term]+1 # increment df for current term
            # merge the current page index with the main index
            index[term].append(posting)
    
            
        # compute idf
        for term in df:
            idf[term] = np.round(np.log(float(num_tweets/df[term])),4)
            
    return index, tf, df, idf

def create_tweets_dict(data):
    '''
    Create a dictionary of key: tweet_id, value: text
    '''
    tweetsDict = {}
    for tweet in data:
        try: text = tweet["full_text"]
        except: text = tweet["text"]
        tweetsDict[tweet["id_str"]] = text
    return tweetsDict

def rankDocuments(query_terms, docs, index, idf, tf, tweetsDict):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    query_terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    
    Returns:
    Print the list of ranked documents
    """
    
    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaing elements would became 0 when multiplied to the queryVector
    docVectors=defaultdict(lambda: [0]*len(query_terms)) # I call docVectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    queryVector=[0]*len(query_terms)

    # compute the norm for the query tf
    query_terms_count = Counter(query_terms) # get the frequency of each term in the query.
    
    query_norm = la.norm(list(query_terms_count.values()))
    
    for termIndex, term in enumerate(query_terms): #termIndex is the index of the term in the query
        if term not in index:
            continue
                    
        ## Compute tf*idf(normalize tf as done with documents)
        queryVector[termIndex]=query_terms_count[term]/query_norm * idf[term]

        # Generate docVectors for matching docs
        for docIndex, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                docVectors[doc][termIndex]=tf[term][docIndex] * idf[term]

    # calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]
    docScores.sort(reverse=True)

    # Get tweet_ids and return them
    resultDocs=[x[1] for x in docScores]
    resultScore = [x[0] for x in docScores]
    if len(resultDocs) == 0:
        return None, None
        
    return resultDocs, resultScore
    
    
def search_tf_idf(query, index, tf, idf, tweetsDict):
    '''
    output is the list of documents that contain any of the query terms.
    So, we will get the list of documents for each query term, and take the union of them.
    '''
    query = normalize_text(query)
    docs = [set()] * len(query)
    for i, term in enumerate(query):
        try:
            # store in termDocs the ids of the docs that contain "term"
            termDocs=[posting[0] for posting in index[term]]
            
            # docs = docs Union termDocs
            docs[i] = set(termDocs)
        except:
            #term is not in index
            pass
        
    docs = list(set.intersection(*docs))
    ranked_docs, ranked_scores = rankDocuments(query, docs, index, idf, tf, tweetsDict)
    
    return ranked_docs, ranked_scores

def mean_w2v(text, w2v_model, tweet = False):
    '''
    return the mean vector created from the embedded matrix of the word2vec model
    '''
    # Get normalized text
    if tweet: n_text = get_normalized_tweet(text)
    else: n_text = normalize_text(text)
    
    # Compute W2V embedding for each term
    w2v_vectors = []
    for term in n_text:
        w2v_vectors.append(w2v_model.wv[term])
    
    if len(w2v_vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    
    # Compute mean vector
    final_vector = np.zeros(w2v_model.vector_size)
    for vec in w2v_vectors:
        for pos in range(w2v_model.vector_size):
            final_vector[pos] += vec[pos]
            
    return final_vector/len(w2v_vectors)

def personalized_rank(query, index, tf, idf, tweetsDict, data, lamb = 1/3):
    '''
    the output is a dataframe for each ranked tweet with 3 columns:
    tf-idf (which is the value based on tf-idf score)
    partial (which is the value computed with the tweet's attributes such as favorite_count, retweeted and user's followers_count)
    personalized (which is the value that comes from the linear combination of the two scores before)
    '''
    # GET TF-IDF SCORE
    ranked_docs, ranked_scores = search_tf_idf(query, index, tf, idf, tweetsDict)
    if ranked_docs == None:
        return None
    
    df_tweets = pd.DataFrame.from_records(data)
    
    # GET WORD2VEC SCORE
    # Create model
    words = []
    for tweet in data:
        try: text = tweet["full_text"]
        except: text = tweet["text"]
        words.append(normalize_text(text))
        
    w2v_model = Word2Vec(sentences = words, size = 100, window = 10, min_count = 0, negative = 10, sg = 0)
    
    # Create mean vectors
    query_tweets = df_tweets[df_tweets["id_str"].isin([str(d) for d in ranked_docs])]
    query_vector = mean_w2v(query, w2v_model)
    
    # create score
    w2v_scores = [[np.dot(query_vector, mean_w2v(row[1], w2v_model, tweet=True)), row[1]["id_str"]] for row in query_tweets.iterrows()]
    w2v_df = pd.DataFrame(w2v_scores, columns = ["w2v_score", "tweet_id"])
    
    # GET PERSONALIZED SCORE
    # Compute partial score (taking into account favorite_count, retweeted and user's followers_count)
    # Get contextual information of each document containing all query terms
    query_tweets = df_tweets[df_tweets["id_str"].isin([str(d) for d in ranked_docs])][["id_str", "user", "favorite_count", "retweeted"]]
    query_tweets["followers"] = [row["followers_count"] for row in query_tweets["user"].values]
    query_tweets.drop(columns = ["user"], inplace = True)
    query_tweets["retweeted"] = query_tweets["retweeted"].apply(lambda x: 1 if x == True else 0)

    # Normalize dimensions -> map to [0, 1]
    max_followers = query_tweets["followers"].max(axis = 0)
    max_favorites = query_tweets["favorite_count"].max(axis = 0)
    if max_followers > 0: query_tweets["followers"] = query_tweets["followers"]/max_followers
    if max_favorites > 0: query_tweets["favorite_count"] = query_tweets["favorite_count"]/max_favorites

    # Get query_vector (best possible combination of attributes)
    query_vector = [1, 0, 1]
    
    # Compute cosine similarity between query_vector and doc_vector
    partial_scores = [[np.dot(query_vector, row[1][["favorite_count", "retweeted", "followers"]].values), row[1]["id_str"]] for row in query_tweets.iterrows()]
    df1 = pd.DataFrame(partial_scores, columns = ["partial_score", "tweet_id"])
    df2 = pd.DataFrame({"tf-idf_score":ranked_scores, "tweet_id":ranked_docs})
    
    # Compute final score (personalized)
    scores_dict = {}  # Dict to store all scores
    max_tfidf = df2["tf-idf_score"].max(axis=0)
    
    for id_str in df1["tweet_id"]:
        # Get partial scores
        tfidf_score = df2[df2["tweet_id"] == int(id_str)]["tf-idf_score"].values[0]
        partial_score = df1[df1["tweet_id"] == id_str]["partial_score"].values[0]
        # Personalized score
        final_score = tfidf_score + lamb * max_tfidf * partial_score
        # W2V score
        w2v_score = w2v_df[w2v_df["tweet_id"] == id_str]["w2v_score"].values[0]
        # Create dict with all the scores
        scores_dict[id_str] = [tfidf_score, partial_score, final_score, w2v_score]

    return pd.DataFrame.from_dict(scores_dict, orient='index', columns=["tf-idf", "partial", "personalized", "w2v"])
    
    
def display_ranking(scores_df, tweetsDict, query, score = "personalized", top_k = 10):
    '''
    print the top-k retrieved tweets from a given query sorted by the chosen score
    score options: "tf-idf", "partial", "personalized", "w2v"
    '''
    print(bcolors.GREEN + "\nTop-" + str(top_k) + " documents sorted by " + score + " score" + bcolors.ENDC)
    
    n_query = normalize_text(query)
    
    scores_df = scores_df.sort_values(by=score, ascending = False)
    for row in scores_df[0:top_k].iterrows():
        string = bcolors.DIM + "{}:".format(row[0]) + bcolors.ENDC
        for term in tweetsDict[str(row[0])].split():
            n_term = normalize_text(term)
            if len(n_term) > 0 and n_term[0] in n_query:
                string += bcolors.MAGENTA + " {}".format(term) + bcolors.ENDC
            else:
                string += " {}".format(term)
        print(string + "\n")
