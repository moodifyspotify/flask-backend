import pymongo
import ssl
from urllib.parse import quote_plus as quote
import pandas as pd
import numpy as np


class MongoConnector:
    def __init__(self, host, user, password, rs, db, auth_db):
        url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
            user=quote(user),
            pw=quote(password),
            hosts=','.join([
                host
            ]),
            rs=rs,
            auth_src=auth_db)
        self.dbs = pymongo.MongoClient(url,
                                       ssl_ca_certs='CA.crt',
                                       ssl_cert_reqs=ssl.CERT_REQUIRED)[db]

    def create_spotify_user(self, email, name, spotyfy_info, auth_info):
        result = self.dbs.users.update_one({'email': email},
                                           {
                                               '$setOnInsert': {
                                                   'email': email,
                                                   'name': name,
                                                   'spotify_info': {
                                                       'user': spotyfy_info,
                                                       'auth': auth_info
                                                   },
                                                   'track_history': [],
                                                   'mood_history': []
                                               }
                                           }, upsert=True)
        return result

    def get_user_by(self, field, value):
        result = self.dbs.users.find_one({
            field: value
        })
        return result

    def update_user(self, by_field, by_value, new_values_dict):
        result = self.dbs.users.update_one({by_field: by_value},
                                           {'$set': new_values_dict})
        return result

    def create_spotify_track(self, track_id, source_name, track_name, artist_name,emotions,lyrics):
        result = self.dbs.tracks.update_one({'track_id': track_id},
                                           {
                                               '$setOnInsert': {
                                                    'track_id': track_id,
                                                    'source_name': source_name,
                                                    'track_name': track_name,
                                                    'artist_name': artist_name,
                                                    'emotions': emotions,
                                                    'lyrics': lyrics
                                               }
                                           }, upsert=True)
        return result

    def get_mood_history_as_pandas(self, email):
        res = list(self.dbs.users.find({'email': email},
                                       {'mood_history': 1, '_id': 0}))
        if len(res) == 0:
            return None
        res_df_init = pd.DataFrame.from_dict(res[0]['mood_history'], orient='index')
        features = ['calm', 'energetic', 'happy', 'sad']
        res_df = pd.DataFrame(res_df_init.music.tolist(), index=res_df_init.index, columns=features)
        res_df.index = pd.to_datetime(res_df.index)
        res_df['date'] = res_df.index.date
        is_f = []
        for i, f in enumerate(features):
            is_f.append('is_' + f)
            res_df['is_' + f] = res_df[features].apply(lambda x: 1 if np.argmax(x) == i else 0, axis=1)
        ret_df = res_df[['date'] + features].groupby('date').mean()
        ret_df[is_f] = res_df[['date'] + is_f].groupby('date').sum()
        ret_df['date'] = ret_df.index
        return ret_df

    def check_processed_tracks(self, tracks):
        result = self.dbs.tracks.find({
            'track_id': {'$in': tracks}
        })

        processed_tracks = {}
        for i in result:
            processed_tracks[i['track_id']] = i

        return processed_tracks

