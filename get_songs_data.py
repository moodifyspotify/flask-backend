from yandex_music import Client
from datetime import datetime
from pathlib import Path
import time
from extract_music_features import extract_feature
import shutil
import pandas as pd
import pickle
import numpy as np
import uuid
import json
from json import JSONEncoder


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return obj.astype(float)
        elif isinstance(obj, np.float16):
            return obj.astype(float)
        elif isinstance(obj, np.float64):
            return obj.astype(float)
        elif isinstance(obj, np.int32):
            return obj.astype(int)
        elif isinstance(obj, np.int64):
            return obj.astype(int)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return JSONEncoder.default(self, obj)


class SongProcessing:
    def __init__(self,token):
        self.token = token
        client = Client.from_token(token)
        self.ya_client = client
        self.uid = str(client.me.account.uid)

    def get_tracks_full_info(self,user_history,tracks_count = 100):

        def download_track(track,track_id,folder):
            track_path = folder + '/' + str(track_id).replace(':','_') + '.mp3'
            track.download(filename=track_path)
            return track_path

        def get_lyrics(track):
            if track.get_supplement()['lyrics'] != None:
                return track.get_supplement()['lyrics']['full_lyrics'].replace('\n', ' ')
            else:
                return None


            # if track_id in list(df['track_id']):
            #     return df[df['track_id'] == track_id]['lyrics'][0], df
            # else:
            #     if track.get_supplement()['lyrics'] != None:
            #         lyrics = track.get_supplement()['lyrics']['full_lyrics'].replace('\n', ' ')
            #     else:
            #         lyrics = None
            #
            #     df = df.append({'track_id': track_id, 'lyrics': lyrics}, ignore_index=True)
            #
            #     return lyrics, df


        user_history_w_lyrics = {}
        tracks_info = self.ya_client.tracks(list(user_history.values())[:tracks_count])

        mp3savepath = 'mp3s/{0}'.format(str(int(time.time()))+str(self.uid))
        self.mp3savepath = mp3savepath
        Path(self.mp3savepath).mkdir(parents=True, exist_ok=True)

        for k,track in enumerate(tracks_info):

            lyrics = get_lyrics(track)

            track_path = download_track(track, list(user_history.values())[k], self.mp3savepath)

            user_history_w_lyrics[list(user_history.keys())[k]] = {
                'track_id': list(user_history.values())[k],
                'track_name': track['title'],
                'track_duration': track['duration_ms'],
                'track_lyrics': lyrics,
                'file_path': track_path
            }


        return user_history_w_lyrics


    def get_music_features(self):
        features_set = extract_feature(self.mp3savepath+'/')
        print(features_set)
        shutil.rmtree(self.mp3savepath)
        return features_set

    def get_lyrics_emotions(self,sd_model,texts):

        fn = f'dl/data/data{uuid.uuid4()}.csv'
        with open(fn, 'w') as f:
            f.write('text\n')
            for t in texts:
                if t != None:
                    f.write(t.replace("\n", " ")+"\n")

        classes = np.round(sd_model.classify(fn)[1], 2)
        res = []
        classes_res_id = 0

        for t in texts:
            if t != None:
                res.append(list(classes[classes_res_id]))
                classes_res_id = classes_res_id + 1
            else:
                res.append([0.0]*8)

        return res


    def get_music_emotions(self,music_features):
        loaded_model = pickle.load(open('dl/music_classifier_knn/knnpickle_file', 'rb'))
        result = loaded_model.predict(music_features)
        return result

    def get_user_songs_history(self,numberOfTracks=1):
        def get_history_formatted(history,numberOfTracks):
            res = {}
            for context in history['contexts']:
                for track in context['tracks']:
                    if 'album_id' in track['track_id']:
                        res[datetime.timestamp(datetime.strptime(track['timestamp'], '%Y-%m-%dT%H:%M:%S%z')) + 3600 * 3] = \
                        "{0}:{1}".format(str(track['track_id']['id_']),str(track['track_id']['album_id']))
            return dict(sorted(res.items(),reverse=True)[:numberOfTracks])

        req_str = 'https://api.music.yandex.net/users/{0}/contexts?types=album,artist,playlist&contextCount=30'.format(self.uid)
        user_history = get_history_formatted(self.ya_client.request.get(req_str),numberOfTracks)
        return user_history

    @staticmethod
    def get_user_stats(access_token, num_tracks, sd_model):
        with open('songs_files/songs_info.json', 'r') as f:
            songs_info = json.load(f)

        sngs = SongProcessing(access_token)
        hist = sngs.get_user_songs_history(num_tracks)
        hist_processed = {}
        temp_i = hist.copy().items()
        for timestamp, track_id in temp_i:
            if track_id in songs_info:
                hist_processed[timestamp] = songs_info[track_id]
                del hist[timestamp]
        del temp_i

        final_res_user = []

        if len(hist) > 0:
            hist_w_lyrics = sngs.get_tracks_full_info(hist, num_tracks)
            feat = sngs.get_music_features()
            emotions = sngs.get_music_emotions(feat[[r for r in feat.columns if r != 'song_name']])
            emotions_lyrics = sngs.get_lyrics_emotions(sd_model,
                                                       [l['track_lyrics'] for l in list(hist_w_lyrics.values())])

            for k, track_id in enumerate(hist_w_lyrics):
                print(type(k))
                final_res_user.append({
                    'timestamp': datetime.utcfromtimestamp(track_id).strftime('%Y-%m-%d'),
                    'is_angry_music': int(emotions[k] == 'angry'),
                    'is_happy_music': int(emotions[k] == 'happy'),
                    'is_sad_music': int(emotions[k] == 'sad'),
                    'is_relaxed_music': int(emotions[k] == 'relaxed'),
                    'anger_lyrics': emotions_lyrics[k][0],
                    'anticipation_lyrics': emotions_lyrics[k][1],
                    'disgust_lyrics': emotions_lyrics[k][2],
                    'fear_lyrics': emotions_lyrics[k][3],
                    'joy_lyrics': emotions_lyrics[k][4],
                    'sadness_lyrics': emotions_lyrics[k][5],
                    'surprise_lyrics': emotions_lyrics[k][6],
                    'trust_lyrics': emotions_lyrics[k][7]
                })

                songs_info[hist_w_lyrics[track_id]['track_id']] = {
                    'track_name': hist_w_lyrics[track_id]['track_name'],
                    'music_emotion': emotions[k],
                    'lyrics_emotion': emotions_lyrics[k]
                }

        for hist_track in hist_processed:
            final_res_user.append({
                'timestamp': datetime.utcfromtimestamp(hist_track).strftime('%Y-%m-%d'),
                'is_angry_music': int(hist_processed[hist_track]['music_emotion'] == 'angry'),
                'is_happy_music': int(hist_processed[hist_track]['music_emotion'] == 'happy'),
                'is_sad_music': int(hist_processed[hist_track]['music_emotion'] == 'sad'),
                'is_relaxed_music': int(hist_processed[hist_track]['music_emotion'] == 'relaxed'),
                'anger_lyrics': hist_processed[hist_track]['lyrics_emotion'][0],
                'anticipation_lyrics': hist_processed[hist_track]['lyrics_emotion'][1],
                'disgust_lyrics': hist_processed[hist_track]['lyrics_emotion'][2],
                'fear_lyrics': hist_processed[hist_track]['lyrics_emotion'][3],
                'joy_lyrics': hist_processed[hist_track]['lyrics_emotion'][4],
                'sadness_lyrics': hist_processed[hist_track]['lyrics_emotion'][5],
                'surprise_lyrics': hist_processed[hist_track]['lyrics_emotion'][6],
                'trust_lyrics': hist_processed[hist_track]['lyrics_emotion'][7]
            })

        df_to_charts = pd.DataFrame(final_res_user).groupby('timestamp').agg(
            {
                'is_angry_music': 'sum',
                'is_happy_music': 'sum',
                'is_sad_music': 'sum',
                'is_relaxed_music': 'sum',
                'anger_lyrics': 'mean',
                'anticipation_lyrics': 'mean',
                'disgust_lyrics': 'mean',
                'fear_lyrics': 'mean',
                'joy_lyrics': 'mean',
                'sadness_lyrics': 'mean',
                'surprise_lyrics': 'mean',
                'trust_lyrics': 'mean'
            }
        )

        df_to_charts.fillna(0, inplace=True)
        df_to_charts.reset_index(inplace=True)

        def get_main_emotion(x):
            lyrics_emtions_list = [
                'anger',
                'anticipation',
                'disgust',
                'fear',
                'joy',
                'sadness',
                'surprise',
                'trust',
            ]
            music_emotions_list = [
                'anger',
                'joy',
                'sadness',
                'trust'
            ]
            lyrics_values = list(x[5:])
            if max(lyrics_values) > 0:
                return lyrics_emtions_list[lyrics_values.index(max(lyrics_values))]
            else:
                music_values = list(x[1:5])
                return music_emotions_list[music_values.index(max(music_values))]

        df_to_charts['main_mood'] = df_to_charts.apply(get_main_emotion, axis=1)

        final_chart_json = {}
        for i in df_to_charts.columns:
            if i not in ['timestamp', 'main_mood']:
                final_chart_json[i] = list(df_to_charts[i].values.astype(float))
            else:
                final_chart_json[i] = list(df_to_charts[i].values)

        with open('songs_files/songs_info.json', 'w', encoding='utf-8') as f:
            json.dump(songs_info, f, ensure_ascii=False, indent=3, cls=NumpyEncoder)

        return final_chart_json
