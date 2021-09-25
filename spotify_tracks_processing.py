from keras.models import model_from_json
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

class MusicClassification:
    def __init__(self):
        self.model = model_from_json(open('dl/music_classifier_keras/music_classifier.json').read())
        self.model.load_weights('dl/music_classifier_keras/model_weights.h5')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.scaler = joblib.load('dl/music_classifier_keras/scaler.save')

    def get_music_emotions(self,tracks_features):
        ids = []
        features = []
        for k,v in tracks_features.items():
            ids.append(k)
            features.append([
                 v['duration_ms'],
                 v['danceability'],
                 v['acousticness'],
                 v['energy'],
                 v['instrumentalness'],
                 v['liveness'],
                 v['valence'],
                 v['loudness'],
                 v['speechiness'],
                 v['tempo']
            ])
        predicts = self.model.predict(self.scaler.transform(features))

        return dict(zip(ids,list(map(list,predicts))))

class LyricsProcessing():

    def __init__(self,genius_api_token):
        self.genius_api_token = genius_api_token
        # 'NFtV-3Xxz9bcZ4Xo_9bfy7LKqrAhSTATV78SO3udcqHr1np-XZZmt53t3_ZS69X8'

    @staticmethod
    def request_song_info(track_name, track_artist,genius_api_token):
        def find_exact(response, track_artist):
            json = response.json()
            remote_song_info = None
            for hit in json['response']['hits']:
                if track_artist.lower() in hit['result']['primary_artist']['name'].lower():
                    remote_song_info = hit
                    break
            return remote_song_info

        base_url = 'https://api.genius.com'
        headers = {'Authorization': 'Bearer ' + genius_api_token}
        search_url = base_url + '/search'
        data = {'q': track_name + ' ' + track_artist}
        response = requests.get(search_url, params=data, headers=headers)
        exact_match = find_exact(response, track_artist)
        return exact_match

    @staticmethod
    def scrape_lyrics(song_url):
        page = requests.get(song_url)
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics1 = html.find("div", class_="lyrics")
        lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
        if lyrics1:
            lyrics = lyrics1.get_text()
        elif lyrics2:
            lyrics = lyrics2.get_text()
        elif lyrics1 == lyrics2 == None:
            lyrics = None
        return lyrics

    def get_lyrics(self, tracks):
        lyrics = {}
        for trid, trinf in tracks.items():
            si = self.request_song_info(trinf['track_name'], trinf['artist_name'], self.genius_api_token)
            if si is None:
                lyrics[trid] = None
            else:
                lyrics[trid] = self.scrape_lyrics(si['result']['url'])
        return lyrics
