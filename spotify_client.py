import json
import requests
import urllib
import base64
import datetime
import time


class SpotifyAuthClient:

    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.base_url = 'https://accounts.spotify.com'
        self.auth_url = '/authorize'
        self.token_url = '/api/token'

    def get_auth_url(self):

        params = '?'+urllib.parse.urlencode({
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'show_dialog': 'true',
            'scope': 'user-read-private user-read-email user-read-currently-playing user-library-read user-read-recently-played'
        })

        return self.base_url+self.auth_url+params

    @staticmethod
    def get_token(code, client_id, client_secret, redirect_uri, grant_type):
        base_url = 'https://accounts.spotify.com'
        token_url = '/api/token'

        b_enc_secr = base64.b64encode(bytes(f"{client_id}:{client_secret}", "ISO-8859-1")). \
            decode("ascii")
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {b_enc_secr}'
        }
        code_type = 'code'
        if grant_type == 'refresh_token':
            code_type = 'refresh_token'
        query = {
            'grant_type': grant_type,
            code_type: code,
            'redirect_uri': redirect_uri
        }
        query = urllib.parse.urlencode(query)

        resp = requests.post(base_url + token_url, data=query, headers=headers)
        rj = resp.json()
        if 'access_token' not in rj:
            raise Exception('Failed to log in.')
        rj['issued_dt'] = str(datetime.datetime.utcnow())
        return rj

    def get_user_token(self, code):
        return SpotifyAuthClient.get_token(code,
                                    self.client_id,
                                    self.client_secret,
                                    self.redirect_uri,
                                    'authorization_code')

    def refresh_user_token(self, code):
        return SpotifyAuthClient.get_token(code,
                                           self.client_id,
                                           self.client_secret,
                                           self.redirect_uri,
                                           'refresh_token')


class SpotifyUserClient:

    def __init__(self, access_info, client_id, client_secret, redirect_uri):
        self.access_info = access_info
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        token_issued_dt = datetime.datetime.strptime(access_info['issued_dt'], '%Y-%m-%d %H:%M:%S.%f')
        if (datetime.datetime.utcnow() - token_issued_dt).seconds > access_info['expires_in']-10:
            auth_clt = SpotifyAuthClient(client_id, client_secret, redirect_uri)
            self.access_info = auth_clt.refresh_user_token(access_info['refresh_token'])

        self.base_api_url = 'https://api.spotify.com/v1'

    def refresh_access(self):
        token_issued_dt = datetime.datetime.strptime(self.access_info['issued_dt'], '%Y-%m-%d %H:%M:%S.%f')
        print(self.access_info)
        if (datetime.datetime.utcnow() - token_issued_dt).seconds > self.access_info['expires_in']-10:
            auth_clt = SpotifyAuthClient(self.client_id, self.client_secret, self.redirect_uri)
            self.access_info = auth_clt.refresh_user_token(self.access_info['refresh_token'])

    def get_user_info(self):
        method_url = '/me'
        self.refresh_access()
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Authorization': f'Bearer {self.access_info["access_token"]}'
        }

        resp = requests.get(self.base_api_url + method_url, headers=headers)
        print(resp.text)
        rj = resp.json()
        return rj

    def get_user_recent_tracks(self):
        method_url = '/me/player/recently-played'

        last_day_tracks = {}

        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Authorization': f'Bearer {self.access_info["access_token"]}'
        }

        resp = requests.get(self.base_api_url + method_url,
                            params={
                                    'after': int(time.time() * 1000)-40*60*1000,
                                    'limit': 50
                                },
                            headers=headers).json()
        for lu in resp['items']:
            last_day_tracks[lu['played_at']] = {
                    'track_name': lu['track']['name'],
                    'artist_names': [artist['name'] for artist in lu['track']['artists']],
                    'track_id': lu['track']['uri']
            }

        return last_day_tracks

    def get_tracks_features(self,track_ids):
        method_url = '/audio-features'
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Authorization': f'Bearer {self.access_info["access_token"]}'
        }

        resp = requests.get(self.base_api_url + method_url,
                             params={
                                 'ids': ','.join(list(set(track_ids)))
                             },
                             headers=headers)
        tracks_audio_features = {}
        for track in resp.json()['audio_features']:
            tracks_audio_features[track['id']] = track

        return tracks_audio_features


class SpotifyAppClient:

    def __init__(self, user_token, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.base_url = 'https://accounts.spotify.com'
        self.auth_url = '/authorize'
        self.token_url = '/api/token'

    def get_auth_url(self):
        params = '?' + urllib.parse.urlencode({
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'show_dialog': 'true',
            'scope': 'user-read-private user-read-email user-read-currently-playing user-library-read'
        })

        return self.base_url + self.auth_url + params

    def get_user_token(self, code):
        b_enc_secr = base64.b64encode(bytes(f"{self.client_id}:{self.client_secret}", "ISO-8859-1")). \
            decode("ascii")
        headers = {
            'Content-type': 'application/x-www-form-urlencoded',
            'Authorization': f'Basic {b_enc_secr}'
        }
        query = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.redirect_uri
        }
        query = urllib.parse.urlencode(query)

        resp = requests.post(self.base_url + self.token_url, data=query, headers=headers)
        rj = resp.json()


