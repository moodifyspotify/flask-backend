import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def extract_feature(path):
    def get_features_for_track(path):
        try:
            y, sr = librosa.load(path, duration=60)
            S = np.abs(librosa.stft(y))
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            # Extracting Features
            tasks = [
                (librosa.beat.beat_track, dict(y=y, sr=sr)),
                (librosa.feature.chroma_stft, dict(y=y, sr=sr)),
                (librosa.feature.chroma_cqt, dict(y=y, sr=sr)),
                (librosa.feature.chroma_cens, dict(y=y, sr=sr)),
                (librosa.feature.melspectrogram, dict(y=y, sr=sr)),
                (librosa.feature.rms, dict(y=y)),
                (librosa.feature.spectral_centroid, dict(y=y, sr=sr)),
                (librosa.feature.spectral_bandwidth, dict(y=y, sr=sr)),
                (librosa.feature.spectral_contrast, dict(S=S, sr=sr)),
                (librosa.feature.spectral_rolloff, dict(y=y, sr=sr)),
                (librosa.feature.poly_features, dict(S=S, sr=sr)),
                (librosa.feature.tonnetz, dict(y=y, sr=sr)),
                (librosa.feature.zero_crossing_rate, dict(y=y)),
                (librosa.effects.harmonic, dict(y=y)),
                (librosa.effects.percussive, dict(y=y)),
                (librosa.feature.delta, dict(data=mfcc)),
                (librosa.frames_to_time, dict(frames=onset_frames[:20], sr=sr))
            ]

            def process_task(task):
                return task[0](**task[1])

            with ProcessPoolExecutor(4) as l_executor:
                results = l_executor.map(process_task, tasks)
            print(results)
            tempo, beats = results[0]  # librosa.beat.beat_track(y=y, sr=sr)
            chroma_stft = results[1]  # librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cq = results[2]  # librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_cens = results[3]  # librosa.feature.chroma_cens(y=y, sr=sr)
            melspectrogram = results[4]  # librosa.feature.melspectrogram(y=y, sr=sr)
            rmse = results[5]  # librosa.feature.rms(y=y)
            cent = results[6]  # librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = results[7]  # librosa.feature.spectral_bandwidth(y=y, sr=sr)
            contrast = results[8]  # librosa.feature.spectral_contrast(S=S, sr=sr)
            rolloff = results[9]  # librosa.feature.spectral_rolloff(y=y, sr=sr)
            poly_features = results[10]  # librosa.feature.poly_features(S=S, sr=sr)
            tonnetz = results[11]  # librosa.feature.tonnetz(y=y, sr=sr)
            zcr = results[12]  # librosa.feature.zero_crossing_rate(y=y)
            harmonic = results[13]  # librosa.effects.harmonic(y=y)
            percussive = results[14]  # librosa.effects.percussive(y=y)
            mfcc_delta = results[15]  # librosa.feature.delta(mfcc)
            frames_to_time = results[16]  # librosa.frames_to_time(onset_frames[:20], sr=sr)

            # Transforming Features
            return [path,  # song name
                    tempo,  # tempo
                    sum(beats),  # beats
                    np.average(beats),
                    np.mean(chroma_stft),  # chroma stft
                    np.std(chroma_stft),
                    np.var(chroma_stft),
                    np.mean(chroma_cq),  # chroma cq
                    np.std(chroma_cq),
                    np.var(chroma_cq),
                    np.mean(chroma_cens),  # chroma cens
                    np.std(chroma_cens),
                    np.var(chroma_cens),
                    np.mean(melspectrogram),  # melspectrogram
                    np.std(melspectrogram),
                    np.var(melspectrogram),
                    np.mean(mfcc),  # mfcc
                    np.std(mfcc),
                    np.var(mfcc),
                    np.mean(mfcc_delta),  # mfcc delta
                    np.std(mfcc_delta),
                    np.var(mfcc_delta),
                    np.mean(rmse),  # rmse
                    np.std(rmse),
                    np.var(rmse),
                    np.mean(cent),  # cent
                    np.std(cent),
                    np.var(cent),
                    np.mean(spec_bw),  # spectral bandwidth
                    np.std(spec_bw),
                    np.var(spec_bw),
                    np.mean(contrast),  # contrast
                    np.std(contrast),
                    np.var(contrast),
                    np.mean(rolloff),  # rolloff
                    np.std(rolloff),
                    np.var(rolloff),
                    np.mean(poly_features),  # poly features
                    np.std(poly_features),
                    np.var(poly_features),
                    np.mean(tonnetz),  # tonnetz
                    np.std(tonnetz),
                    np.var(tonnetz),
                    np.mean(zcr),  # zero crossing rate
                    np.std(zcr),
                    np.var(zcr),
                    np.mean(harmonic),  # harmonic
                    np.std(harmonic),
                    np.var(harmonic),
                    np.mean(percussive),  # percussive
                    np.std(percussive),
                    np.var(percussive),
                    np.mean(frames_to_time),  # frames
                    np.std(frames_to_time),
                    np.var(frames_to_time)]
        except:
            return [0.0]*55

    # Traversing over each file in path
    file_data = [f for f in listdir(path) if isfile(join(path, f))]
    res = []
    file_data = map(lambda line: path + line[:-1] if line[-1:] == '\n' else path + line, file_data)

    with ThreadPoolExecutor(30) as executor:
        exec_result = executor.map(get_features_for_track, file_data)

    for i in exec_result:
        res.append(i)

    return pd.DataFrame(res, columns=['song_name', 'tempo', 'total_beats', 'average_beats',
                                      'chroma_stft_mean', 'chroma_stft_std', 'chroma_stft_var',
                                      'chroma_cq_mean', 'chroma_cq_std', 'chroma_cq_var', 'chroma_cens_mean',
                                      'chroma_cens_std', 'chroma_cens_var', 'melspectrogram_mean',
                                      'melspectrogram_std', 'melspectrogram_var', 'mfcc_mean', 'mfcc_std',
                                      'mfcc_var', 'mfcc_delta_mean', 'mfcc_delta_std', 'mfcc_delta_var',
                                      'rmse_mean', 'rmse_std', 'rmse_var', 'cent_mean', 'cent_std',
                                      'cent_var', 'spec_bw_mean', 'spec_bw_std', 'spec_bw_var',
                                      'contrast_mean', 'contrast_std', 'contrast_var', 'rolloff_mean',
                                      'rolloff_std', 'rolloff_var', 'poly_mean', 'poly_std', 'poly_var',
                                      'tonnetz_mean', 'tonnetz_std', 'tonnetz_var', 'zcr_mean', 'zcr_std',
                                      'zcr_var', 'harm_mean', 'harm_std', 'harm_var', 'perc_mean', 'perc_std',
                                      'perc_var', 'frame_mean', 'frame_std', 'frame_var'])