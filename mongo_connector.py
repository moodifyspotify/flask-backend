import pymongo
import ssl
from urllib.parse import quote_plus as quote


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


