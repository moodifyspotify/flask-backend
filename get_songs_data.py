from yandex_music import Client
from datetime import datetime
from pathlib import Path
import time
from extract_music_features import extract_feature
import shutil
import pandas as pd
import pickle

class SongProcessing:
    def __init__(self,token):
        self.token = token
        client = Client.from_token(token)
        self.ya_client = client
        self.uid = str(client.me.account.uid)

    def get_tracks_full_info(self,user_history,tracks_count = 100):

        def download_track(track,track_id,folder):
            track_path = folder + '/' + str(track_id) + '.mp3'
            track.download(filename=track_path)
            return track_path

        def get_lyrics(track,track_id,df):
            if track_id in df['track_id']:
                return df[df['track_id'] == track_id]['lyrics']
            else:
                if track.get_supplement()['lyrics'] != None:
                    lyrics = track.get_supplement()['lyrics']['full_lyrics'].replace('\n', ' ')
                else:
                    lyrics = None

                df.append({'track_id': track_id, 'lyrics': lyrics}, ignore_index=True)

                return lyrics, df


        user_history_w_lyrics = {}
        tracks_info = self.ya_client.tracks(list(user_history.values())[:tracks_count])

        mp3savepath = '/Users/roma/Desktop/flask-mood/mp3s/{0}'.format(str(time.time())+str(self.uid))
        self.mp3savepath = mp3savepath
        Path(self.mp3savepath).mkdir(parents=True, exist_ok=True)
        songs_w_lyrics = pd.read_csv('/Users/roma/Desktop/flask-mood/songs_files/lyrics.csv')

        for id,track in enumerate(tracks_info):

            lyrics,songs_w_lyrics = get_lyrics(
                track, list(user_history.values())[id], songs_w_lyrics
            )

            track_path = download_track(track, list(user_history.values())[id], self.mp3savepath)

            user_history_w_lyrics[list(user_history.keys())[id]] = {
                'track_id': list(user_history.values())[id],
                'track_name': track['title'],
                'track_duration': track['duration_ms'],
                'track_lyrics': lyrics,
                'file_path': track_path
            }

        songs_w_lyrics.to_csv('/Users/roma/Desktop/flask-mood/songs_files/lyrics.csv', index=None)

        return user_history_w_lyrics


    def get_music_features(self):
        features_set = extract_feature(self.mp3savepath+'/')
        shutil.rmtree(self.mp3savepath)
        return features_set

    def get_lyrics_emotions(lyrics):
        return None

    def get_music_emotions(music_features):
        loaded_model = pickle.load(open('/Users/roma/Desktop/flask-mood/dl/music_classifier_knn/knnpickle_file', 'rb'))
        result = loaded_model.predict(music_features)
        return result

    def get_user_songs_history(self):
        def get_history_formatted(history):
            res = {}
            for context in history['contexts']:
                for track in context['tracks']:
                    res[datetime.timestamp(datetime.strptime(track['timestamp'], '%Y-%m-%dT%H:%M:%S%z')) + 3600 * 3] = \
                    "{0}:{1}".format(str(track['track_id']['id_']),str(track['track_id']['album_id']))
            return dict(sorted(res.items(),reverse=True))

        req_str = 'https://api.music.yandex.net/users/{0}/contexts?types=album,artist,playlist&contextCount=30'.format(self.uid)
        user_history = get_history_formatted(self.ya_client.request.get(req_str))
        return user_history
