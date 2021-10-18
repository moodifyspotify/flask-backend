import os
from dl.models import SentimentDiscovery
from mongo_connector import MongoConnector
import numpy as np
import uuid

import logging

sd_model = SentimentDiscovery()

mongo_conn = MongoConnector('rc1a-zptn64g6pn8ylwgh.mdb.yandexcloud.net:27018',
                            'mood_user',
                            'MoodGfhjkm_017',
                            'rs01', 'mood', 'mood')

logging.basicConfig(filename='job_logs.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s : %(message)s')


def job():
    logging.info('Job started')
    non_processed_tracks = mongo_conn.get_non_processed_lyrics()

    non_processed_texts = list(map(lambda x: x['lyrics']['text'], non_processed_tracks))
    logging.info(f'{len(non_processed_texts)} texts to analyze')

    for i, text in enumerate(non_processed_texts):
        try:
            logging.info(f'{i} text started ({non_processed_tracks[i]["track_name"]})')
            fn = f'dl/data/data{uuid.uuid4()}.csv'
            with open(fn, 'w', encoding='utf-8') as f:
                f.write('text\n')
                f.write('"' + text.replace("\n", " ").replace('"', '\\"') + '"\n')
            classes = np.round(sd_model.classify(fn)[1], 2)[0]
            track_emotions = non_processed_tracks[i]['emotions']
            track_emotions['lyrics'] = [float(x) for x in classes]
            mongo_conn.update_track('_id', non_processed_tracks[i]['_id'], {'emotions': track_emotions})
            print('Done:', non_processed_tracks[i]['track_name'], track_emotions['lyrics'])
            os.remove(fn)
        except Exception as e:
            logging.error(f'{i} text error ({non_processed_tracks[i]["track_name"]})')
            logging.error(e)


if __name__ == "__main__":
    job()
