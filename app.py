from fastapi import FastAPI
import snscrape.modules.twitter as sntwitter
import re    # RegEx for removing non-letter characters
import pickle
from keras.utils import pad_sequences
from keras.models import load_model
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

app = FastAPI()

load_dotenv()
API_KEY=os.getenv('youtube_api_key_cred')
youtube = build("youtube","v3",developerKey=API_KEY)

tokenizer=None
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sentiment_model=load_model('./models/best_model.h5')
sentiments=['Negative','Neutral','Positive']

def tweet_to_words(tweet):
    ''' Convert tweet text into a sequence of words '''
    max_len=100
    # convert to lowercase
    text = tweet.lower()
    text = re.sub(r"\S*https?:\S*", "", text)
    # remove non letters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    text=[text]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, padding='post', maxlen=max_len)
    result=sentiment_model.predict(text).argmax(1)
    result=sentiments[result[0]]
    return result


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/get_twitter_sentiments/{data}")
def get_twitter_sentiments(data):
    topic=data
    hashtag=data
    # query="pathaan (#pathaan) until:2023-01-14 since:2022-12-01"
    # query = topic + " (#" + hashtag + ") " + "until:"+totimeyyyy+"-"+totimemm+"-"+totimeday+ " since:"+fromtimeyyyy+"-"+fromtimemm +"-"+fromtimeday +" lang:en"
    query = topic +" lang:en"
    tweets=[]
    limits=100
    for tweet in sntwitter.TwitterSearchScraper('(from:elonmusk) until:2022-01-01 since:2010-01-01').get_items():
        if len(tweets)==limits:
            break
        else:
            tweets.append(tweet.rawContent)
    # tweets = list(map(tweet_to_words,tweets))
    # print(tweets[0])
    return {'list':"hii"}

@app.get("/api/get_text_sentiments/{data}")
def get_text_sentiments(data):
    text=data
    print(data)
    result = tweet_to_words(text)
    # print(tweets[0])
    return {'result':result}


@app.post("/api/get_youtube_sentiments/")
def get_youtube_sentiments(req):
    s=req
    # print(s)
    if s.find('watch') == -1:
        s=s.split('/')[-1]
    else:
        s=re.sub(r'watch','',s)
        s=re.sub(r'[^\w]v','=',s)
        s=s.split('=')
        s=s[2].split('&')[0]

    # print(s)
    request=youtube.commentThreads().list(
        part='id,replies,snippet',
        order='relevance',
        videoId=s,
        maxResults=100
    )
    response=request.execute()
    com_list=[]
    for i in response['items']:
        com_list.append(i['snippet']['topLevelComment']['snippet']['textOriginal'])
    
    result = list(map(tweet_to_words,com_list))
    return {'result':result}
