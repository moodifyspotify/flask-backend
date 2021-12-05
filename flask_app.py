from flask import Flask, jsonify, request, render_template, session, redirect, g, flash, url_for, make_response
from flask_apscheduler import APScheduler
from flask_cors import CORS, cross_origin
import requests
import urllib
import base64

from spotify_client import SpotifyAuthClient, SpotifyUserClient, SpotifyAppClient
from spotify_tracks_processing import MusicClassification, LyricsProcessing
import numpy as np

import random

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px

import json
from json import JSONEncoder

from mongo_connector import MongoConnector

import logging

# client_id = 'none'
# client_secret = 'none'

logging.basicConfig(filename='logs.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s : %(message)s')


sp_client_id = '3561e398cf0e414da717da295a2c0e91'
sp_client_secret = '7f7503a4c32e4878926a23f0eb06aaec'
if __name__ == "__main__":
    sp_redirect_uri = 'http://172.20.10.3:5000/spotify_auth'
else:
    sp_redirect_uri = 'https://mude.ml/spotify_auth'

sp_client = SpotifyAuthClient(sp_client_id, sp_client_secret, sp_redirect_uri)

test_data = {"anger_lyrics": [0.0, 0.0, 0.4400000050663948, 0.0, 0.0, 0.07000000153978665],
             "anticipation_lyrics": [0.0, 0.0, 0.0, 0.0, 0.0, 0.009999999776482582],
             "disgust_lyrics": [0.0, 0.0, 0.8350000083446503, 0.0, 0.0, 0.07444444422920544],
             "fear_lyrics": [0.15000000596046448, 0.0, 0.07500000018626451, 0.0, 0.0, 0.0],
             "is_angry_music": [0.0, 0.0, 0.0, 2.0, 0.0, 1.0], "is_happy_music": [0.0, 2.0, 2.0, 3.0, 1.0, 6.0],
             "is_relaxed_music": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "is_sad_music": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             "joy_lyrics": [0.6899999976158142, 0.0, 0.0, 0.0, 0.0, 0.17222221692403158],
             "main_mood": ["joy", "joy", "disgust", "joy", "joy", "joy"],
             "sadness_lyrics": [0.009999999776482582, 0.0, 0.635000005364418, 0.0, 0.0, 0.005555555431379212],
             "surprise_lyrics": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             "timestamp": ["2021-07-30", "2021-07-31", "2021-08-06", "2021-08-08", "2021-08-09", "2021-08-21"],
             "trust_lyrics": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


class Config:
    SCHEDULER_API_ENABLED = True


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


def get_test_plot(data,reply_history):
    songs_count = sum(data['is_calm_music']) + \
                  sum(data['is_energetic_music']) + \
                  sum(data['is_happy_music']) + \
                  sum(data['is_sad_music'])

    data['day_sn'] = data['is_calm_music'] + data['is_energetic_music'] + data['is_happy_music'] + data['is_sad_music']
    data['calm_perc'] = data['is_calm_music'] / data['day_sn']
    data['energetic_perc'] = data['is_energetic_music'] / data['day_sn']
    data['happy_perc'] = data['is_happy_music'] / data['day_sn']
    data['sad_perc'] = data['is_sad_music'] / data['day_sn']


    data['Спокойствие'] = data['trust_lyrics'] + 0.25 * data['calm_perc']
    data['Ярость'] = data['anger_lyrics']
    data['Восторг'] = data['anticipation_lyrics'] + 0.25 * data['energetic_perc']
    data['Веселье'] = data['joy_lyrics'] + 0.25 * data['happy_perc']
    data['Неприязнь'] = data['disgust_lyrics']
    data['Страх'] = data['fear_lyrics']
    data['Удивление'] = data['surprise_lyrics']
    data['Грусть'] = data['sadness_lyrics'] + 0.25 * data['sad_perc']

    emts = ['Спокойствие', 'Ярость', 'Восторг', 'Веселье', 'Неприязнь', 'Страх', 'Удивление', 'Грусть']

    def get_main_emotion(x):
        vls = list(x[emts])
        return emts[vls.index(max(vls))]

    data['main_mood'] = data.apply(get_main_emotion, axis=1)
    data[emts] = data[emts].div(data[emts].sum(axis=1),axis=0)
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

    v_map_reply = {
        'mood_terrible': -2,
        'mood_bad': -1,
        'mood_ok': 0,
        'mood_good': 1,
        'mood_excellent': 2,
    }


    sms = [data[e].sum() for e in emts]
    sms = list(map(lambda x: x / sum(sms), sms))
    pie_df = pd.DataFrame({
        "Настроение": emts,
        "Величина": sms
    })

    st_b_d = {'Дата': [], 'Настроение': [], 'Величина': []}
    l_d = {'Дата': [], 'Настроение': [], 'Величина': [], 'z':[]}

    for ts in data['date']:
        k = data[data['date'] == ts]['main_mood'].values[0]
        v = v_map[k]
        l_d['Дата'].append(ts)
        l_d['Настроение'].append(k)
        l_d['Величина'].append(v)
        l_d['z'].append(5)

        for e in emts:
            st_b_d['Дата'].append(ts)
            st_b_d['Настроение'].append(e)
            st_b_d['Величина'].append(data[data['date'] == ts][e].values[0])


    bar_df = pd.DataFrame(st_b_d)

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
    cdm_reply = {
        'mood_terrible': '#FF6F76',
        'mood_bad': '#FFCA2D',
        'mood_ok': '#6DCE8A',
        'mood_good': '#8D92A5',
        'mood_excellent': '#97F3FD'
    }




    res_values = []
    res_labels = []

    bar_datasets = []
    v_map_rev = dict((v, k) for k, v in v_map.items())
    v_map_reply_rev = dict((v, k) for k, v in v_map_reply.items())


    pie_cfg = {
        'type': 'pie',
        'data': {
            'labels': list(pie_df['Настроение']),
            'datasets': [{
                'data': list(pie_df['Величина']),
                'backgroundColor': list(map(lambda x: cdm[x],list(pie_df['Настроение']))),
                'hoverOffset': 4
            }]
        },
        'options': {
            'responsive': True,
            'plugins': {
                'legend': {
                    'display': False,
                },
            },
        },
    }

    line_cfg = {
        'type': 'line',
        'data': {
            'labels': list(map(str,l_d['Дата'])),
            'datasets': [{
                'lineTension': 0.5,
                'data': l_d['Величина'],
                'pointBackgroundColor': list(map(lambda x: cdm[v_map_rev[x]],l_d['Величина']))
            }]
        },
        'options': {
             'responsive': True,
            'maintainAspectRatio': False,
            'borderWidth': 12,
            'elements': {
                'point': {
                    'radius': 7,
                    'borderWidth': 8,
                    'hoverBorderWidth': 8
                }
            },
            'bezierCurve': True,
            'plugins': {
                'legend': {
                    'display': False,
                },
                'tooltip': {
                    'displayColors': False
                }
            },
            'scales': {
                'x': {
                    'type': 'timeseries',
                    'time': {
                        'unit': 'day'
                    },
                    'grid': {
                        'display': False,
                        'drawBorder': False
                    },
                    'ticks': {
                        'color': '#FFFFFF',
                        'source': 'data'
                    }

                },
                'y': {
                    'display': False,
                    'grid': {
                        'display': False
                    }
                },
            },
        },
    }

    line_cfg_reply = dict(line_cfg)

    line_cfg_reply['data'] = {
            'labels': list(map(str,reply_history.index)),
            'datasets': [{
                'lineTension': 0.5,
                'data': list(map(lambda x: v_map_reply[x], reply_history.values)),
                'pointBackgroundColor': list(map(lambda x: cdm_reply[x],reply_history.values))
            }]
        }


    gradient_set = []
    gradient_set_reply = []
    def get_gradient_scale(gradient_set):
        res = []
        for i,val in enumerate(gradient_set):
            if i == 0:
                res.append([i,val[1]])
            else:
                res.append([i/(len(gradient_set)-1), val[1]])
        return res


    for i in sorted(list(set(line_cfg['data']['datasets'][0]['data']))):
        gradient_set.append([i,cdm[v_map_rev[i]]])
    gradient_set = get_gradient_scale(gradient_set)

    for i in sorted(list(set(line_cfg_reply['data']['datasets'][0]['data']))):
        gradient_set_reply.append([i, cdm_reply[v_map_reply_rev[i]]])
    gradient_set_reply = get_gradient_scale(gradient_set_reply)


    bar_cfg = {
        'type': 'bar',
        'data': {},
        'options': {
            'plugins':{
                'legend': {
                    'display': False
                }
            },
            'responsive': True,
            'maintainAspectRatio': False,
            'categoryPercentage': 1.0,
            'barPercentage': 1.0,
            'scales': {
                'x': {
                    'type': 'timeseries',
                    'time': {
                        'unit': 'day'
                    },
                    'stacked': True,
                    'grid': {
                        'display': False,
                        'drawBorder': False
                    },
                    'ticks': {
                        'color': '#FFFFFF',
                        'source': 'data'
                    }

                },
                'y': {
                    'stacked': True,
                    'display': False,
                    'grid': {
                        'display': False
                    }
                },
            },

        }
    }

    for i in bar_df['Настроение'].unique():
        bar_datasets.append(
            {
                'label': i,
                'data': list(bar_df[bar_df['Настроение'] == i].sort_values('Дата')['Величина']),
                'backgroundColor': cdm[i]
            }
        )
        res_labels.append(i)
        res_values.append(list(bar_df[bar_df['Настроение'] == i].sort_values('Дата')['Величина']))

    bar_cfg['data'] = {
        'labels': list(map(str,bar_df[bar_df['Настроение'] == i].sort_values('Дата')['Дата'])),
        'datasets': bar_datasets
    }




    return json.dumps(line_cfg, ensure_ascii=False),\
           json.dumps(bar_cfg, ensure_ascii=False),\
           json.dumps(gradient_set, ensure_ascii=False),\
           json.dumps(pie_cfg, ensure_ascii=False), \
           sorted(list(zip(pie_cfg['data']['labels'], pie_cfg['data']['datasets'][0]['data'], pie_cfg['data']['datasets'][0]['backgroundColor'])),
                  key=lambda x:x[1], reverse=True),\
           json.dumps(gradient_set_reply, ensure_ascii=False),\
            json.dumps(line_cfg_reply, ensure_ascii=False),





def get_plots():
    return


def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    app.secret_key = 'rand' + str(random.random())
    app.config.from_object(Config())

    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    mongo_conn = MongoConnector('rc1a-zptn64g6pn8ylwgh.mdb.yandexcloud.net:27018',
                                'mood_user',
                                'MoodGfhjkm_017',
                                'rs01', 'mood', 'mood')

    @scheduler.task('cron', id='history_scarp', second=0, minute='*/5')
    def scarp_users_history():
        users = mongo_conn.get_all_users()
        mc = MusicClassification()
        for u in users:
            try:
                logging.info(f'Scarping music of: {u["name"]}')
                access_info = u['spotify_info']['auth']
                sp_user_clt = SpotifyUserClient(access_info, sp_client_id, sp_client_secret, sp_redirect_uri)
                tracks_history = sp_user_clt.get_user_recent_tracks()
                tracks_to_process = [i['track_id'] for i in tracks_history.values()]
                processed_tracks,processed_tracks_full = mongo_conn.check_processed_tracks(tracks_to_process)
                unprocessed_tracks_ids = [i for i in tracks_to_process if i not in list(processed_tracks.keys())]

                for i in processed_tracks_full:
                    users_listened = i.get('users_listened', [])
                    if u['spotify_info']['user']['id'] not in users_listened:
                        users_listened.append(u['spotify_info']['user']['id'])
                        mongo_conn.update_track('track_id', 'users_listened', users_listened)

                if len(unprocessed_tracks_ids) > 0:
                    track_features = sp_user_clt.get_tracks_features([i.split(':')[-1] for i in unprocessed_tracks_ids])
                    classes = mc.get_music_emotions(track_features)
                    lp = LyricsProcessing('NFtV-3Xxz9bcZ4Xo_9bfy7LKqrAhSTATV78SO3udcqHr1np-XZZmt53t3_ZS69X8')
                    to_l = {}
                    for l in tracks_history.values():
                        if l['track_id'] in unprocessed_tracks_ids:
                            to_l[l['track_id']] = {
                                'track_name': l['track_name'],
                                'artist_name': l['artist_names'][0]
                            }
                    lyrics = lp.get_lyrics(to_l)
                user_info_history = u['track_history']
                user_info_mood_history = u['mood_history']

                for track_timestamp, track_info in tracks_history.items():
                    ins = dict(track_id=track_info['track_id'],
                               source_name='spotify',
                               track_name=track_info['track_name'],
                               artist_name=track_info['artist_names'],
                               emotions={
                                   'music': list(map(float, classes[track_info['track_id'].split(':')[-1]])) \
                                       if track_info['track_id'] in unprocessed_tracks_ids \
                                       else processed_tracks[track_info['track_id']]['emotions']['music']
                               },
                               lyrics={
                                   'text': lyrics[track_info['track_id']] \
                                       if track_info['track_id'] in unprocessed_tracks_ids \
                                       else processed_tracks[track_info['track_id']]['lyrics']
                               },
                               users_listened=[u['spotify_info']['user']['id']])
                    if track_info['track_id'] in unprocessed_tracks_ids:
                        mongo_conn.create_spotify_track(**ins)

                    del ins['lyrics']
                    user_info_history[track_timestamp] = ins
                    user_info_mood_history[track_timestamp] = ins['emotions']

                mongo_conn.update_user('email', u['email'], {
                    'track_history': user_info_history,
                    'mood_history': user_info_mood_history
                })

            except Exception as e:
                logging.error(str(e))

    @app.route('/login', methods=['POST', 'GET'])
    @cross_origin()
    def login_tg():
        if request.method == "GET":
            tg_id = request.args.get('tg_id')
            token = request.args.get('token')
            if not tg_id:
                return redirect('/')
            if not token:
                return redirect('/')
            if not mongo_conn.check_tg_token(int(tg_id), token):
                return redirect('/')

            resp = make_response(redirect('/'))
            resp.set_cookie('tg_access', json.dumps({'tg_id': int(tg_id)}), max_age=60 * 60 * 24 * 365 * 2)
            at = request.cookies.get('access_info')
            if at:
                return resp

            try:
                link = sp_client.get_auth_url()
                resp = make_response(render_template('login.html', spotify_auth_link=link))
                resp.set_cookie('tg_access', json.dumps({'tg_id': int(tg_id)}), max_age=60 * 60 * 24 * 365 * 2)
                return resp
            except Exception as e:
                resp = make_response(redirect('/'))
                resp.set_cookie('tg_access', json.dumps({'tg_id': int(tg_id)}), max_age=60 * 60 * 24 * 365 * 2)
                return resp

    @app.route('/')
    def main_page():
        at = request.cookies.get('access_info')
        if at:
            # if request.args.get('n'):
            #     num_tracks = int(request.args.get('n'))
            access_info = json.loads(at)
            sp_user_clt = SpotifyUserClient(access_info, sp_client_id, sp_client_secret, sp_redirect_uri)
            try:
                user_info = sp_user_clt.get_user_info()
                data,reply_history = mongo_conn.get_mood_history_as_pandas(user_info['email'])
                print(data)
                if data is None:
                    data = test_data
                    flash('Показываем тестовых рыбов')

                line_cfg, bar_cfg, gradient_set, pie_cfg, legend,gradient_set_reply,line_cfg_reply = get_test_plot(data,reply_history)

                g.user = {
                    'username': user_info['display_name'],
                    'access_token': sp_user_clt.access_info['access_token']
                }

                resp = make_response(render_template('notdash.html',
                                                     bar_cfg=bar_cfg,
                                                     line_cfg=line_cfg,
                                                     gradient_set=gradient_set,
                                                     pie_cfg=pie_cfg,
                                                     legend_list=legend,
                                                     gradient_set_reply=gradient_set_reply,
                                                     line_cfg_reply=line_cfg_reply
                                                     ))
                return resp
            except Exception as e:
                print(str(e))
                flash('Что-то пошло не так( Попробуйте зайти снова')
                link = sp_client.get_auth_url()
                return render_template('login.html', spotify_auth_link=link)
        else:
            link = sp_client.get_auth_url()
            return render_template('login.html', spotify_auth_link=link)

    @app.route('/spotify_auth', methods=['POST', 'GET'])
    @cross_origin()
    def spoti_auth():
        if request.method == "GET":
            error = request.args.get('error')
            # tg = request.cookies.get('tg_access')
            # if not tg:
            #     return redirect('/')
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
                # tg_id = json.loads(tg)['tg_id']
                # mongo_conn.add_spotify_info(tg_id, user_info['email'], user_info['uri'], user_info, token)
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
            tracks_to_process = [i['track_id'] for i in tracks_history.values()]
            processed_tracks = mongo_conn.check_processed_tracks(tracks_to_process)
            unprocessed_tracks_ids = [i for i in tracks_to_process if i not in list(processed_tracks.keys())]

            if len(unprocessed_tracks_ids) > 0:
                track_features = sp_user_clt.get_tracks_features([i.split(':')[-1] for i in unprocessed_tracks_ids])
                mc = MusicClassification()
                classes = mc.get_music_emotions(track_features)
                lp = LyricsProcessing('NFtV-3Xxz9bcZ4Xo_9bfy7LKqrAhSTATV78SO3udcqHr1np-XZZmt53t3_ZS69X8')
                to_l = {}
                for l in tracks_history.values():
                    if l['track_id'] in unprocessed_tracks_ids:
                        to_l[l['track_id']] = {
                            'track_name': l['track_name'],
                            'artist_name': l['artist_names'][0]
                        }
                lyrics = lp.get_lyrics(to_l)

            spoti_email = sp_user_clt.get_user_info()['email']
            user_info = mongo_conn.get_user_by('email', spoti_email)
            user_info_history = user_info['track_history']
            user_info_mood_history = user_info['mood_history']

            for track_timestamp, track_info in tracks_history.items():
                ins = dict(track_id=track_info['track_id'],
                           source_name='spotify',
                           track_name=track_info['track_name'],
                           artist_name=track_info['artist_names'],
                           emotions={
                               'music': list(map(float, classes[track_info['track_id'].split(':')[-1]])) \
                                   if track_info['track_id'] in unprocessed_tracks_ids \
                                   else processed_tracks[track_info['track_id']]['emotions']['music']
                           },
                           lyrics={
                               'text': lyrics[track_info['track_id']] \
                                   if track_info['track_id'] in unprocessed_tracks_ids \
                                   else processed_tracks[track_info['track_id']]['lyrics']
                           })
                if track_info['track_id'] in unprocessed_tracks_ids:
                    mongo_conn.create_spotify_track(**ins)

                del ins['lyrics']
                user_info_history[track_timestamp] = ins
                user_info_mood_history[track_timestamp] = ins['emotions']

            mongo_conn.update_user('email', spoti_email, {
                'track_history': user_info_history,
                'mood_history': user_info_mood_history
            })

            return str(processed_tracks)
        return 'ne work'


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
