from flask import Flask, jsonify, request, render_template, session, redirect, g, flash, url_for, make_response

from flask_cors import CORS, cross_origin
import requests
import urllib
import base64

from spotify_client import SpotifyAuthClient, SpotifyUserClient, SpotifyAppClient
from yandex_music import Client, exceptions
from spotify_tracks_processing import MusicClassification,LyricsProcessing
# from dl.models import SentimentDiscovery

import multiprocessing as mp
import numpy as np

import random
import uuid

import json
from datetime import datetime

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from get_songs_data import SongProcessing
import json
from json import JSONEncoder

from mongo_connector import MongoConnector
import sys
import traceback

client_id = 'none'
client_secret = 'none'

sp_client_id = '3561e398cf0e414da717da295a2c0e91'
sp_client_secret = '7f7503a4c32e4878926a23f0eb06aaec'
if __name__ == "__main__":
    sp_redirect_uri = 'http://192.168.1.65:5000/spotify_auth'
else:
    sp_redirect_uri = 'https://music-mood-tracker.ml/spotify_auth'

sp_client = SpotifyAuthClient(sp_client_id, sp_client_secret, sp_redirect_uri)
#
# sd_model = SentimentDiscovery()

test_data = {  "anger_lyrics": [    0.0,    0.0,    0.4400000050663948,    0.0,    0.0,    0.07000000153978665  ],  "anticipation_lyrics": [    0.0,    0.0,    0.0,    0.0,    0.0,    0.009999999776482582  ],  "disgust_lyrics": [    0.0,    0.0,    0.8350000083446503,    0.0,    0.0,    0.07444444422920544  ],  "fear_lyrics": [    0.15000000596046448,    0.0,    0.07500000018626451,    0.0,    0.0,    0.0  ],  "is_angry_music": [    0.0,    0.0,    0.0,    2.0,    0.0,    1.0  ],  "is_happy_music": [    0.0,    2.0,    2.0,    3.0,    1.0,    6.0  ],  "is_relaxed_music": [    0.0,    0.0,    0.0,    0.0,    0.0,    0.0  ],  "is_sad_music": [    1.0,    0.0,    0.0,    0.0,    0.0,    0.0  ],  "joy_lyrics": [    0.6899999976158142,    0.0,    0.0,    0.0,    0.0,    0.17222221692403158  ],  "main_mood": [    "joy",    "joy",    "disgust",    "joy",    "joy",    "joy"  ],  "sadness_lyrics": [    0.009999999776482582,    0.0,    0.635000005364418,    0.0,    0.0,    0.005555555431379212  ],  "surprise_lyrics": [    0.0,    0.0,    0.0,    0.0,    0.0,    0.0  ],  "timestamp": [    "2021-07-30",    "2021-07-31",    "2021-08-06",    "2021-08-08",    "2021-08-09",    "2021-08-21"  ],  "trust_lyrics": [    0.0,    0.0,    0.0,    0.0,    0.0,    0.0  ]}

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


def get_test_plot(data):
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

    data_df['Спокойствие'] = data_df['trust_lyrics'] + 0.25*data_df['relaxed_music_perc']
    data_df['Ярость'] = data_df['anger_lyrics'] + 0.25*data_df['angry_music_perc']
    data_df['Восторг'] = data_df['anticipation_lyrics']
    data_df['Веселье'] = data_df['joy_lyrics'] + 0.25*data_df['happy_music_perc']
    data_df['Неприязнь'] = data_df['disgust_lyrics']
    data_df['Страх'] = data_df['fear_lyrics']
    data_df['Удивление'] = data_df['surprise_lyrics']
    data_df['Грусть'] = data_df['sadness_lyrics'] + 0.25*data_df['sad_music_perc']

    emts = ['Спокойствие', 'Ярость', 'Восторг', 'Веселье',
            'Неприязнь', 'Страх', 'Удивление', 'Грусть']

    def get_main_emotion(x):
        vls = list(x[emts])
        return emts[vls.index(max(vls))]

    data_df['main_mood'] = data_df.apply(get_main_emotion, axis=1)


    v_map = {
        'Ярость': -3,
        'Страх': -2,
        'Неприязнь': -1,
        'Грусть': 0,
        'Спокойствие': 1,
        'Удивление': 2,
        'Восторг': 3,
        'Веселье': 4
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
        v = v_map[k]
        l_d['Дата'].append(datetime.strptime(ts, '%Y-%m-%d'))
        l_d['Настроение'].append(k)
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
        'Неприязнь': '#6DCE8A',
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
    bar_fig.update_layout(barnorm="percent")

    fig1 = px.line(line_df, x="Дата", y="Величина")
    fig1.update_traces(line=dict(color='#ECF8F7'))
    fig2 = px.scatter(line_df, x='Дата', y='Величина',
                      template='plotly_dark',
                      color='Настроение',
                      color_discrete_map=cdm,
                      size='z'
                      # trendline="rolling",
                      # trendline_options=dict(window=1)
                      )
    line_fig = go.Figure(data=fig1.data + fig2.data)
    line_fig.layout.plot_bgcolor = "#292E43"
    line_fig.layout.paper_bgcolor = "#292E43"
    line_fig.update_yaxes(visible=False)
    line_fig.update_layout(template='plotly_dark')
    line_fig.update_traces(line_shape='spline')

    return json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder), \
           json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder), \
           json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_plots():
    return


def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    app.secret_key = 'rand'+str(random.random())
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    mongo_conn = MongoConnector('rc1a-zptn64g6pn8ylwgh.mdb.yandexcloud.net:27018',
                                'mood_user',
                                'MoodGfhjkm_017',
                                'rs01', 'mood', 'mood')

    @app.route('/')
    def main_page():
        at = request.cookies.get('access_info')
        if at:
            # if request.args.get('n'):
            #     num_tracks = int(request.args.get('n'))
            # else:
            #     num_tracks = 20
            # y_clnt = Client(at)
            # g.user = {
            #         'username': y_clnt.me.account.login,
            #         'access_token': at
            #     }

            # try:
            #     data = SongProcessing.get_user_stats(at,
            #                             num_tracks,
            #                             sd_model)
            # except BaseException as e:
            #     error = "Что-то пошло не так( Показываем тестовых рыбов"
            #
            #     ex_type, ex_value, ex_traceback = sys.exc_info()
            #
            #     # Extract unformatter stack traces as tuples
            #     trace_back = traceback.extract_tb(ex_traceback)
            #
            #     # Format stacktrace
            #     stack_trace = list()
            #
            #     for trace in trace_back:
            #         stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
            #         trace[0], trace[1], trace[2], trace[3]))
            #         flash("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
            #         trace[0], trace[1], trace[2], trace[3]))
            #
            #     data = test_data
            #     flash(str(e))
            #     flash(error)
            #
            #     print("Exception type : %s " % ex_type.__name__)
            #     print("Exception message : %s" % ex_value)
            #     print("Stack trace : %s" % stack_trace)
            #
            #     flash("Exception type : %s " % ex_type.__name__)
            #     flash("Exception message : %s" % ex_value)

            access_info = json.loads(at)
            sp_user_clt = SpotifyUserClient(access_info, sp_client_id, sp_client_secret, sp_redirect_uri)
            try:
                user_info = sp_user_clt.get_user_info()
                data = test_data
                flash('Показываем тестовых рыбов')
                pieJSON, barJSON, lineJSON = get_test_plot(data)

                g.user = {
                    'username': user_info['display_name'],
                    'access_token': sp_user_clt.access_info['access_token']
                }

                resp = make_response(render_template('notdash.html', pieJSON=pieJSON, barJSON=barJSON, lineJSON=lineJSON))
                return resp
            except Exception as e:
                print(str(e))
                flash('Что-то пошло не так( Попробуйте зайти снова')
                link = sp_client.get_auth_url()
                return render_template('login.html', spotify_auth_link=link)
        else:
            link = sp_client.get_auth_url()
            return render_template('login.html', spotify_auth_link=link)

    # @app.route('/get_songs_history')
    # def songs_history():
    #     session['access_token'] = 'AgAAAAAh7Vk7AAG8XtDkZzG_PEYLjGVYMIVdDQE'
    #     if 'access_token' in session:
    #         num_tracks = int(request.args.get('n'))
    #         final_chart_json = SongProcessing.get_user_stats(session['access_token'],
    #                                                          num_tracks,
    #                                                          sd_model)
    #
    #         return final_chart_json, 200
    #     else:
    #         return redirect('/')

    # @app.route('/dash_test', methods=['GET', 'POST'])
    # def notdash():
    #     pieJSON, barJSON = get_test_plot(session['access_token'])
    #     return render_template('notdash.html', pieJSON=pieJSON, barJSON=barJSON)
    #
    # @app.route('/api/get_text_emotions', methods=['POST'])
    # @cross_origin()
    # def text_emotions():
    #     if request.method == "POST":
    #         request_data = request.get_json()
    #         text = request_data['text']
    #         fn = f'dl/data/data{uuid.uuid4()}.csv'
    #         with open(fn, 'w') as f:
    #             f.write('text\n'+text.replace("\n", " "))
    #         res = sd_model.classify(fn)
    #         print(res)
    #         return {'result': str(res)}, 200
    #     return jsonify({
    #                 'statusCode': 400
    #             }), 400
    #
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
    #                 f.write('"'+t.replace("\n", " ").replace('"', '\\"')+'"\n')
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

    @app.route('/spotify_auth', methods=['POST', 'GET'])
    @cross_origin()
    def spoti_auth():
        if request.method == "GET":
            error = request.args.get('error')
            if error:
                return redirect('/')
            try:
                code = request.args.get('code')
                token = sp_client.get_user_token(code)
                print(token)
                resp = make_response(redirect('/'))
                resp.set_cookie('access_info', json.dumps(token), max_age=60 * 60 * 24 * 365 * 2)
                sp_user_clt = SpotifyUserClient(token, sp_client_id, sp_client_secret, sp_redirect_uri)
                user_info = sp_user_clt.get_user_info()
                mongo_conn.create_spotify_user(user_info['email'], user_info['uri'], user_info, token)
                return resp
            except Exception as e:
                error = "Не удалось войти..."
                flash(error)
                link = sp_client.get_auth_url()
                return render_template('login.html', spotify_auth_link=link)

    @app.route('/get_spotify_history', methods=['POST', 'GET'])
    @cross_origin()
    def get_spotify_history():
        at = request.cookies.get('access_info')
        if at:
            access_info = json.loads(at)
            sp_user_clt = SpotifyUserClient(access_info, sp_client_id, sp_client_secret, sp_redirect_uri)
            tracks_history = sp_user_clt.get_user_recent_tracks()
            track_features = sp_user_clt.get_tracks_features([i['track_id'].split(':')[-1] for i in tracks_history.values()])
            mc = MusicClassification()
            classes = mc.get_music_emotions(track_features)
            lp = LyricsProcessing('NFtV-3Xxz9bcZ4Xo_9bfy7LKqrAhSTATV78SO3udcqHr1np-XZZmt53t3_ZS69X8')
            to_l = {}
            for l in tracks_history.values():
                to_l[l['track_id']] = {
                    'track_name': l['track_name'],
                    'artist_name': l['artist_names'][0]
                }
            lyrics = lp.get_lyrics(to_l)

            spoti_email = sp_user_clt.get_user_info()['email']
            user_info = mongo_conn.get_user_by('email', spoti_email)
            user_info_history = user_info['track_history']
            user_info_mood_history = user_info['mood_history']

            for track_timestamp,track_info in tracks_history.items():
                ins = dict(track_id=track_info['track_id'],
                    source_name='spotify',
                    track_name=track_info['track_name'],
                    artist_name=track_info['artist_names'],
                    emotions={
                        'music': list(map(float,classes[track_info['track_id'].split(':')[-1]]))
                    },
                    lyrics={
                        'text': lyrics[track_info['track_id']]
                    })

                mongo_conn.create_spotify_track(**ins)
                user_info_history[track_timestamp] = ins
                user_info_mood_history[track_timestamp] = ins['emotions']

            mongo_conn.update_user('email', spoti_email,{
                'track_history': user_info_history,
                'mood_history': user_info_mood_history
            })

            return str(classes)
        return 'ne work'



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
                    resp = make_response(redirect('/'))
                    resp.set_cookie('access_token', token, max_age=60 * 60 * 24 * 365 * 2)
                    return resp
                except Exception as e:
                    flash(str(e))
                    error = "Не удалось войти... Вероятный диагноз -- неверный пароль("

            flash(error)
            return render_template('login.html')

    @app.route('/logout', methods=['POST', 'GET'])
    @cross_origin()
    def logout():
        session.clear()
        resp = make_response(redirect(url_for('main_page')))
        resp.set_cookie('access_info', '', expires=0)
        return resp

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', debug=True, threaded=True)
