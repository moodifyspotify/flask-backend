from flask import Flask, jsonify, request, render_template, session, redirect
from flask_cors import CORS, cross_origin
import requests
import urllib

from yandex_music import Client

import random

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


def create_app(app_name='YAMOOD_API'):
    app = Flask(app_name)
    app.secret_key = 'rand'+str(random.random())
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route('/')
    def main_page():
        if 'access_token' in session:
            return f'Tell me you are sad without telling me you are sad <br> {session["access_token"]}'
        else:
            return render_template('login.html')

    @app.route('/get_songs_history')
    def songs_history():
        if 'access_token' in session:
            sngs = SongProcessing(session['access_token'])
            return json.dumps(sngs.get_user_songs_history())
        else:
            return redirect('/')

    @app.route('/dash_test',methods=['GET', 'POST'])
    def notdash():
        df = pd.DataFrame({
            "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
            "Amount": [4, 1, 2, 2, 4, 5],
            "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
        })
        fig = px.bar(df, x="Fruit", y="Amount", color="City",    barmode="group")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('notdash.html', graphJSON=graphJSON)

    @app.route('/api', methods=['GET'])
    @cross_origin()
    def default_f():
        if request.method == "GET":
            return {'result':'It Works!'}, 200
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
            token = get_client_from_cred(username, password)
            session['access_token'] = token
            return redirect('/')

        return jsonify({
                    'statusCode': 400
                }), 400

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0',debug=True)
