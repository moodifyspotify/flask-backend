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
import plotly.graph_objects as go
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
    data = {
      "anger_lyrics": [
        0.0,
        0.0,
        0.4400000050663948,
        0.0,
        0.0,
        0.07000000153978665
      ],
      "anticipation_lyrics": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.009999999776482582
      ],
      "disgust_lyrics": [
        0.0,
        0.0,
        0.8350000083446503,
        0.0,
        0.0,
        0.07444444422920544
      ],
      "fear_lyrics": [
        0.15000000596046448,
        0.0,
        0.07500000018626451,
        0.0,
        0.0,
        0.0
      ],
      "is_angry_music": [
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        1.0
      ],
      "is_happy_music": [
        0.0,
        2.0,
        2.0,
        3.0,
        1.0,
        6.0
      ],
      "is_relaxed_music": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "is_sad_music": [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "joy_lyrics": [
        0.6899999976158142,
        0.0,
        0.0,
        0.0,
        0.0,
        0.17222221692403158
      ],
      "main_mood": [
        "joy",
        "joy",
        "disgust",
        "joy",
        "joy",
        "joy"
      ],
      "sadness_lyrics": [
        0.009999999776482582,
        0.0,
        0.635000005364418,
        0.0,
        0.0,
        0.005555555431379212
      ],
      "surprise_lyrics": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "timestamp": [
        "2021-07-30",
        "2021-07-31",
        "2021-08-06",
        "2021-08-08",
        "2021-08-09",
        "2021-08-21"
      ],
      "trust_lyrics": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    }
    songs_count = sum(data['is_angry_music'])+\
        sum(data['is_happy_music']) + \
        sum(data['is_relaxed_music']) + \
        sum(data['is_sad_music'])


    data_df = pd.DataFrame(data)
    data_df['day_sn'] = data_df['is_angry_music'] + data_df['is_happy_music'] + data_df['is_relaxed_music'] + data_df['is_sad_music']
    data_df['angry_music_perc'] = data_df['is_angry_music']/data_df['day_sn']
    data_df['happy_music_perc'] = data_df['is_happy_music'] / data_df['day_sn']
    data_df['relaxed_music_perc'] = data_df['is_relaxed_music'] / data_df['day_sn']
    data_df['sad_music_perc'] = data_df['is_sad_music'] / data_df['day_sn']

    data_df['Спокойствие'] = data_df['trust_lyrics'] + data_df['relaxed_music_perc']
    data_df['Ярость'] = data_df['anger_lyrics'] + data_df['angry_music_perc']
    data_df['Восторг'] = data_df['anticipation_lyrics']
    data_df['Веселье'] = data_df['joy_lyrics'] + data_df['happy_music_perc']
    data_df['Отвращение'] = data_df['disgust_lyrics']
    data_df['Страх'] = data_df['fear_lyrics']
    data_df['Удивление'] = data_df['surprise_lyrics']
    data_df['Грусть'] = data_df['sadness_lyrics'] + data_df['sad_music_perc']

    emts = ['Спокойствие', 'Ярость', 'Восторг', 'Веселье',
            'Отвращение', 'Страх', 'Удивление', 'Грусть']
    v_map = {
        'anger': (-3, 'Ярость'),
        'fear': (-2, 'Страх'),
        'disgust': (-1, 'Отвращение'),
        'sadness': (0, 'Грусть'),
        'trust': (1, 'Спокойствие'),
        'surprise': (2, 'Удивление'),
        'anticipation': (3, 'Восторг'),
        'joy': (4, 'Веселье')
    }

    sms = [data_df[e].sum() for e in emts]
    pie_df = pd.DataFrame({
        "Настроение": emts,
        "Величина": sms
    })

    st_b_d = {'Дата': [], 'Настроение': [], 'Величина': []}
    l_d = {'Дата': [], 'Настроение': [], 'Величина': [], 'z':[]}
    for ts in data['timestamp']:
        k = data_df[data_df['timestamp'] == ts]['main_mood'].values[0]
        v, m = v_map[k]
        l_d['Дата'].append(datetime.strptime(ts, '%Y-%m-%d'))
        l_d['Настроение'].append(m)
        l_d['Величина'].append(v)
        l_d['z'].append(5)

        for e in emts:
            st_b_d['Дата'].append(ts)
            st_b_d['Настроение'].append(e)
            st_b_d['Величина'].append(round(data_df[data_df['timestamp'] == ts][e].values[0], 2))
    bar_df = pd.DataFrame(st_b_d)
    line_df = pd.DataFrame(l_d)

    cdm = {
        'Ярость': '#FF6F76',
        'Страх': '#FFCA2D',
        'Отвращение': '#6DCE8A',
        'Грусть': '#8D92A5',
        'Спокойствие': '#97F3FD',
        'Удивление': '#FDB5B5',
        'Восторг': '#D076FF',
        'Веселье': '#52A3FD'
    }
    color_scale = [
        [0.0, '#FF6F76'],
        [0.125, '#FFCA2D'],
        [0.25, '#6DCE8A'],
        [0.375, '#8D92A5'],
        [0.625, '#97F3FD'],
        [0.75, '#FDB5B5'],
        [0.875, '#D076FF'],
        [1.0, '#52A3FD']
    ]

    pie_fig = px.pie(pie_df, values="Величина", names="Настроение",
                     template='plotly_dark',
                     color="Настроение",
                     color_discrete_map=cdm)
    pie_fig.update_traces(textinfo='none', hoverinfo='label')
    pie_fig.layout.plot_bgcolor = "#292E43"
    pie_fig.layout.paper_bgcolor = "#292E43"

    bar_fig = px.bar(bar_df, x="Дата", y="Величина",
                     template='plotly_dark',
                     color="Настроение",
                     color_discrete_map=cdm)
    bar_fig.layout.plot_bgcolor = "#292E43"
    bar_fig.layout.paper_bgcolor = "#292E43"
    bar_fig.update_yaxes(visible=False)

    line_fig = px.scatter(line_df, x='Дата', y='Величина',
                          template='plotly_dark',
                          color='Настроение',
                          color_discrete_map=cdm,
                          size='z'
                          #trendline="rolling",
                          #trendline_options=dict(window=1)
                          )
    line_fig.layout.plot_bgcolor = "#292E43"
    line_fig.layout.paper_bgcolor = "#292E43"
    line_fig.update_yaxes(visible=False)
    #line_fig.update_traces(line_shape='spline')

    return json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder), \
           json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder), \
           json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_plots():
    return


def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    app.secret_key = 'rand'+str(random.random())
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route('/')
    def main_page():
        session['access_token'] = 'AgAAAAAh7Vk7AAG8XtDkZzG_PEYLjGVYMIVdDQE'
        if 'access_token' in session:
            y_clnt = Client(session['access_token'])
            g.user = {
                'username': y_clnt.me.account.login,
                'access_token': session['access_token']
            }
            pieJSON, barJSON, lineJSON = get_test_plot()
            return render_template('notdash.html', pieJSON=pieJSON, barJSON=barJSON, lineJSON=lineJSON)
        else:
            return render_template('login.html')

    @app.route('/get_songs_history')
    def songs_history():
        session['access_token'] = 'AgAAAAAh7Vk7AAG8XtDkZzG_PEYLjGVYMIVdDQE'
        if 'access_token' in session:
            num_tracks = 1
            final_chart_json = SongProcessing.get_user_stats(session['access_token'],
                                                             num_tracks,
                                                             sd_model)

            return final_chart_json, 200
        else:
            return redirect('/')

    @app.route('/dash_test', methods=['GET', 'POST'])
    def notdash():
        pieJSON, barJSON = get_test_plot()
        return render_template('notdash.html', pieJSON=pieJSON, barJSON=barJSON)

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
