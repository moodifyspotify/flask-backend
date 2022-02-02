import os

class Mongo:
    USER = os.getenv('MUDE_MONGO_USER', 'user')
    PASSWORD = os.getenv('MUDE_MONGO_PWD', 'password')
    HOST = os.getenv('MUDE_MONGO_HOST')
    DB = os.getenv('MUDE_MONGO_DB')
    REPLICA_SET = os.getenv('MUDE_MONGO_RS', 'rs01')


class Spotify:
    CLIENT_ID = os.getenv('MUDE_SPOTIFY_CLIENTID', 'client_id')
    CLIENT_SECRET = os.getenv('MUDE_SPOTIFY_CLIENTSCRT', 'client_secret')


class Genius:
    API_TOKEN = os.getenv('MUDE_GENUIS_APITOKEN', 'api_token')
