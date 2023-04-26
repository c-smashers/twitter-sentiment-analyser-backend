from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
import re    # RegEx for removing non-letter characters
import pickle
from keras.utils import pad_sequences
from keras.models import load_model
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

app = FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    val=sentiment_model.predict(text)
    senti_score=int(val[0][0]*-1000) + int(val[0][2]*1000)
    if senti_score>100:
        senti_score-=int(val[0][1]*1000)
    elif senti_score > -100:
        senti_score=0
    else:
        senti_score+=int(val[0][1]*1000)

    if senti_score>400:
        result="Positive"
    elif senti_score==0:
        result="Neutral"
    elif senti_score > 0:
        result="Patially Positive"
    else:
        result="Negative"
    senti_score=senti_score//100
    result=[result,senti_score]
    return result


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/get_text_sentiments/{data}")
def get_text_sentiments(data):
    text=data
    # print(data)
    result = tweet_to_words(text)
    return {'result':result[0]}


@app.post("/api/get_youtube_sentiments/")
def get_youtube_sentiments(data:dict):
    s=data['value']
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
    n=len(result)
    vals=[x[1] for x in result]
    result=[x[0] for x in result]
    neg_count=(result.count('Negative')*100)//n
    neu_count=(result.count('Neutral')*100)//n
    part_pos_count=(result.count('Patially Positive')*100)//n
    pos_count=100-neg_count-neu_count-part_pos_count
    result=[neg_count,neu_count,part_pos_count,pos_count]

    request = youtube.videos().list(
        part="snippet,statistics",
        id=s
    )
    response=request.execute()
    vDetails={
        'title':response['items'][0]['snippet']['title'],
        'thumbnail':response['items'][0]['snippet']['thumbnails']['medium']['url'],
        'channel':response['items'][0]['snippet']['channelTitle'],
        'commentcount':response['items'][0]['statistics']['commentCount']
    }

    return {'result':result,'values':vals,'vDetails':vDetails}
