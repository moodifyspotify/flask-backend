from yandex_music import Client
from datetime import datetime

class SongProcessing:
    def __init__(self,token):
        self.token = token
        client = Client.from_token(token)
        self.ya_client = client
        self.uid = str(client.me.account.uid)

    def get_track_info(self,track_id):
        return None

    def download_track(self,track_id):
        return None

    def get_music_features(file_path):
        return None

    def get_lyrics_emotions(lyrics):
        return None

    def get_music_emotions(music_features):
        return None

    def get_user_songs_history(self):
        def get_history_formatted(history):
            res = {}
            for context in history['contexts']:
                for track in context['tracks']:
                    res[datetime.timestamp(datetime.strptime(track['timestamp'], '%Y-%m-%dT%H:%M:%S%z')) + 3600 * 3] = \
                    track['track_id']
            print(res)
            return dict(sorted(res.items(),reverse=True))

        req_str = 'https://api.music.yandex.net/users/{0}/contexts?types=album,artist,playlist&contextCount=30'.format(self.uid)
        user_history = get_history_formatted(self.ya_client.request.get(req_str))
        return user_history
