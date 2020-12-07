from utils import *

parser = argparse.ArgumentParser()

mode_help = "(string) You can input 3 different modes when acquiring the data:" + \
            " \n'hydrate' --> listen to incoming tweets and dump them into json files in batches of 100. " + \
            " \n'scrape' --> listen to incoming tweets and dump them into json files in batches of 100. " + \
            " \n'read' --> only read the already captured data stored in json files (DEFAULT)."
            
score_help = "(string) There are 3 potential scores to rank the tweets with:" + \
             " \n'tf-idf' --> which is the value based on tf-idf score. " + \
             " \n'partial' --> which is the value computed with the tweet's attributes such as favorite_count, retweeted and user's followers_count. " + \
             " \n'personalized' --> which is the value that comes from the linear combination of the two scores before (DEFAULT)." + \
             " \n'w2v' --> which is the ranking coming from the Word2Vec embeddings."
             
topK_help = "(integer) Maximum number of top K documents that will be retrieved (DEFAULT = 10)."

sample_help = "(integer) Sample of data to work with."

scrapingsample_help = "(integer) Number (in thousands) of tweets to be scraped if --mode scrape (DEFAULT = 200)"

parser.add_argument("-m", "--mode", help=mode_help)
parser.add_argument("-s", "--score", help=score_help)
parser.add_argument("-tk", "--topK", help=topK_help, type=int)
parser.add_argument("-sm", "--sample", help=sample_help, type=int)
parser.add_argument("-ss", "--scrapingsample", help=scrapingsample_help, type=int)
args = parser.parse_args()
    
config = configparser.ConfigParser()
config.read("config.ini")
credentials = config["CREDENTIALS"]

## access token informations
access_token1 = credentials["access_token_key"]
access_token_secret1 = credentials["access_token_secret"]

consumer_key1 = credentials["consumer_key"]
consumer_secret1 = credentials["consumer_secret"]

auth = OAuthHandler(consumer_key1, consumer_secret1)
auth.set_access_token(access_token1, access_token_secret1)
api = API(auth_handler=auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

mode = args.mode
if (mode != "scrape" and mode != "hydrate" and mode != "read"):
    print(bcolors.CYAN + "The mode '" + str(mode) + "' is not an available option. The mode was switched to 'read'" + bcolors.ENDC)
    mode = "read"
    
score = args.score
if (score != "personalized" and score != "partial" and score != "tf-idf" and score != "w2v"):
    print(bcolors.CYAN + "The score '" + str(score) + "' is not an available option. The score was switched to 'personalized'" + bcolors.ENDC)
    score = "personalized"

keywords = [keyword.strip('"') for keyword in config["KEYWORDS"]["keywords_list"].strip('][').split(', ')]
            
scraping_sample = args.scrapingsample
if scraping_sample == None: scraping_sample = 200

if (mode == "scrape" or mode == "hydrate"): start = time.time()
data = get_tweets(keywords, scraping_sample, mode = mode, data_directory = '../data/')
if mode == "scrape": print(bcolors.GREEN + "Total time taken to scrape {} tweets was {} seconds".format(scraping_sample * 1000, round(time.time() - start, 4)) + bcolors.ENDC)
if mode == "hydrate": print(bcolors.GREEN + "Total time taken to hydrate {} tweets was {} seconds".format(len(data), round(time.time() - start, 4)) + bcolors.ENDC)
print("Total tweets in database: " + bcolors.BOLD + str(len(data)) + bcolors.ENDC)

if args.sample != None and len(data) > args.sample:
    print("Working with a sample of " + bcolors.BOLD + str(args.sample) + bcolors.ENDC + " tweets\n")
    tweetsDict = create_tweets_dict(data[:args.sample])
    index, tf, df, idf = create_index_tfidf(data[:args.sample], args.sample)
    data = data[:args.sample]
else:
    tweetsDict = create_tweets_dict(data)
    index, tf, df, idf = create_index_tfidf(data, len(data))


while(True):
    print(bcolors.UNDERLINE + "ENTER QUERY:" + bcolors.ENDC)
    query = input()
    if query == "--end":
        print("\n\nClosing search engine...")
        break
    
    if len(query.split()) == 0:
        print(bcolors.YELLOW + "Cannot search for an empty string\n" + bcolors.ENDC)
    
    else:
        # obtain a dataframe with tweet_id as index and tf-if_score, partial_score and final_score (personalized)
        scores = personalized_rank(query, index, tf, idf, tweetsDict, data)
        if type(scores) == type(None):
            print(bcolors.YELLOW + "No results found, try again\n" + bcolors.ENDC)
        else:
            if args.topK != None: display_ranking(scores, tweetsDict, query, score = score, top_k = args.topK)
            else: display_ranking(scores, tweetsDict, query, score = score)
        

