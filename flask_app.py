from flask import Flask, jsonify, request, render_template, session, redirect, g, flash, url_for
from flask_cors import CORS, cross_origin
import requests
import urllib

from yandex_music import Client, exceptions
from dl.models import SentimentDiscovery

import multiprocessing as mp
import numpy as np

import random
import uuid

import json
from datetime import datetime
import time
import ssl
import base64
import secrets

import pandas as pd
import plotly
import plotly.express as px
from get_songs_data import SongProcessing
import json
from json import JSONEncoder

client_id = '0a5c1ff2ba7e4bdd83ee228720efacb5'
client_secret = 'ab444188a16e471cbbdd48965449dff3'

sd_model = SentimentDiscovery()


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


def get_test_plot():
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    app.secret_key = 'rand'+str(random.random())
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route('/')
    def main_page():
        if 'access_token' in session:
            y_clnt = Client(session['access_token'])
            g.user = {
                'username': y_clnt.me.account.login,
                'access_token': session['access_token']
            }
            graphJSON = get_test_plot()
            return render_template('notdash.html', graphJSON=graphJSON)
        else:
            return render_template('login.html')

    @app.route('/get_songs_history')
    def songs_history():
        session['access_token'] = 'AgAAAAAh7Vk7AAG8XtDkZzG_PEYLjGVYMIVdDQE'
        if 'access_token' in session:
            with open('songs_files/songs_info.json', 'r') as f:
                songs_info = json.load(f)

            num_tracks = 400

            sngs = SongProcessing(session['access_token'])
            hist = sngs.get_user_songs_history(num_tracks)
            hist_processed = {}
            temp_i = hist.copy().items()
            for timestamp,track_id in temp_i:
                if track_id in songs_info:
                    hist_processed[timestamp] = songs_info[track_id]
                    del hist[timestamp]
            del temp_i

            final_res_user = []

            if len(hist) > 0:
                hist_w_lyrics = sngs.get_tracks_full_info(hist, num_tracks)
                feat = sngs.get_music_features()
                emotions = sngs.get_music_emotions(feat[[r for r in feat.columns if r != 'song_name']])
                emotions_lyrics = sngs.get_lyrics_emotions(sd_model, [l['track_lyrics'] for l in list(hist_w_lyrics.values())])

                for k,track_id in enumerate(hist_w_lyrics):
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
                    'anticipation_lyrics':  'mean',
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

            df_to_charts['main_mood'] = df_to_charts.apply(get_main_emotion,axis=1)

            final_chart_json = {}
            for i in df_to_charts.columns:
                if i not in ['timestamp', 'main_mood']:
                    final_chart_json[i] = list(df_to_charts[i].values.astype(float))
                else:
                    final_chart_json[i] = list(df_to_charts[i].values)

            with open('songs_files/songs_info.json', 'w', encoding='utf-8') as f:
                json.dump(songs_info, f, ensure_ascii=False, indent=3, cls=NumpyEncoder)

            return final_chart_json, 200
        else:
            return redirect('/')

    @app.route('/dash_test',methods=['GET', 'POST'])
    def notdash():
        graphJSON = get_test_plot()
        return render_template('notdash.html', graphJSON=graphJSON)

    @app.route('/api/get_text_emotions', methods=['POST'])
    @cross_origin()
    def text_emotions():
        if request.method == "POST":
            request_data = request.get_json()
            text = request_data['text']
            fn = f'dl/data/data{uuid.uuid4()}.csv'
            with open(fn, 'w') as f:
                f.write('text\n'+text.replace("\n", " "))
            res = sd_model.classify(fn)
            print(res)
            return {'result': str(res)}, 200
        return jsonify({
                    'statusCode': 400
                }), 400

    # @app.route('/api/get_text_emotions_batch', methods=['POST'])
    # @cross_origin()
    # def text_emotions_batch():
    #     if request.method == "POST":
    #         request_data = request.get_json()
    #         texts = request_data['texts']
    #         fn = f'dl/data/data{uuid.uuid4()}.csv'
    #         with open(fn, 'w') as f:
    #             f.write('text\n')
    #             for t in texts:
    #                 f.write(t.replace("\n", " ")+"\n")
    #         res = np.round(sd_model.classify(fn)[1], 2)
    #
    #         print(res)
    #         return {'result': str(res)}, 200
    #     return jsonify({
    #         'statusCode': 400
    #     }), 400

    def get_client(code):
        token_auth_uri = f"https://oauth.yandex.ru/token"
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
        }
        query = {
            'grant_type': 'authorization_code',
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret,
        }
        query = urllib.parse.urlencode(query)

        resp = requests.post(token_auth_uri, data=query, headers=headers)
        print(resp.text)
        rj = resp.json()
        return rj['access_token']

    def get_client_from_cred(un, pwd):
        return Client.from_credentials(un, pwd).token

    @app.route('/auth', methods=['POST', 'GET'])
    @cross_origin()
    def auth():
        if request.method == "GET":
            code = request.args.get('code')
            token = get_client(code)
            session['access_token'] = token
            return redirect('/')
        elif request.method == "POST":
            username = request.form.get('username')
            password = request.form.get('password')

            error = None

            if not username:
                error = 'Введите логин'
            elif not password:
                error = 'Введите пароль'
            if error is None:
                try:
                    token = get_client_from_cred(username, password)
                    session['access_token'] = token
                    return redirect('/')
                except exceptions.BadRequest:
                    error = "Неудалось войти... Вероятный диагноз -- неверный пароль("

            flash(error)

            return render_template('login.html')

    @app.route('/logout', methods=['POST', 'GET'])
    @cross_origin()
    def logout():
        session.clear()
        return redirect(url_for('main_page'))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', debug=True, threaded=True)
