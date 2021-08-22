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

client_id = '0a5c1ff2ba7e4bdd83ee228720efacb5'
client_secret = 'ab444188a16e471cbbdd48965449dff3'

sd_model = SentimentDiscovery()


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
                'username': y_clnt.me.account.login
            }
            graphJSON = get_test_plot()
            return render_template('notdash.html', graphJSON=graphJSON)
        else:
            return render_template('login.html')

    @app.route('/get_songs_history')
    def songs_history():
        if 'access_token' in session:
            sngs = SongProcessing(session['access_token'])
            hist = sngs.get_user_songs_history()
            hist_w_lyrics = sngs.get_tracks_full_info(hist,10)
            feat = sngs.get_music_features()
            emotions = sngs.get_music_emotions(feat)
            return str(emotions)
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

    @app.route('/api/get_text_emotions_batch', methods=['POST'])
    @cross_origin()
    def text_emotions_batch():
        if request.method == "POST":
            request_data = request.get_json()
            texts = request_data['texts']
            fns = []
            for t in texts:
                fn = f'dl/data/data{uuid.uuid4()}.csv'
                with open(fn, 'w') as f:
                    f.write('text\n' + t.replace("\n", " "))
                fns.append(fn)
            with mp.Pool(2) as pool:
                results = pool.map(sd_model.classify, fns)

            print(results)
            results = list(map(lambda x: np.round(x[1], 2), results))
            return {'result': str(results)}, 200
        return jsonify({
            'statusCode': 400
        }), 400

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

    @app.route('/auth', methods=['POST','GET'])
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
