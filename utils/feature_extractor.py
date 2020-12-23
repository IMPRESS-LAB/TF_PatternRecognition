import os
import librosa
import numpy as np
import pandas as pd

from itertools import chain
from config.config import get_config

config = get_config()


def parse_metadata(path):
    meta = []
    meta_df = pd.read_csv(path)
    meta_df = meta_df[["slice_file_name", "fold", "classID"]]

    for f in range(config.n_fold):
        class_data = meta_df[meta_df.fold == f+1]
        meta.append(list(zip(class_data["slice_file_name"], class_data["fold"], class_data["classID"])))
    return meta


class FEATURE_EXTRACTOR():
    def __init__(self):
        self.n_fold = config.n_fold
        self.sampling_rate = config.sr
        self.n_fft = config.n_fft
        self.mel_dim = config.mel
        self.filter = config.filter
        self.mfc_dim = config.mfc
        self.hop_length = config.hop_len
        self.win_length = config.win_len

    def create_folder(self):
        os.makedirs(config.load_path + '/mel', exist_ok=True)
        [os.makedirs(config.load_path + '/mel/fold' + str(k + 1), exist_ok=True) for k in range(self.n_fold)]
        os.makedirs(config.load_path + '/mfc', exist_ok=True)
        [os.makedirs(config.load_path + '/mfc/fold' + str(k + 1), exist_ok=True) for k in range(self.n_fold)]

    def get_mel(self, file):
        S, _ = librosa.load(file, sr=self.sampling_rate)
        mel = librosa.feature.melspectrogram(S,
                                             sr=self.sampling_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.mel_dim,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length)
        feature = librosa.power_to_db(mel, ref=np.max)

        return feature

    def get_mfcc(self, file):
        S, _ = librosa.load(file, sr=self.sampling_rate)
        mel = librosa.feature.melspectrogram(S,
                                             sr=self.sampling_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.filter,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length)
        log_S = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(S=log_S,
                                    n_mfcc=self.mfc_dim,
                                    sr=self.sampling_rate,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length
                                    )
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        mfcc_delta2 = librosa.feature.delta(mfcc, width=3, order=2)

        feature = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

        return feature


if __name__ == "__main__":
    DATA_PATH = config.load_path
    metadata = parse_metadata(DATA_PATH + "/UrbanSound8K.csv")
    wavPath = DATA_PATH + '/wav'
    extractor = FEATURE_EXTRACTOR()
    
    extractor.create_folder()   # 특징을 저장할 폴더 생성 (./data/mel/fold#, ./data/mfc/fold#)
    fileName, fileFold, fileLabel = zip(*list(chain(*metadata)))

    for idx, val in enumerate(fileName):
        read_wav = f"{wavPath}/fold{fileFold[idx]}/{val}"
        
        # 특징 추출
        mel_feat = extractor.get_mel(read_wav)  # (n_mel, n_frames)
        mfc_feat = extractor.get_mfcc(read_wav) # (n_mfc*2, n_frames)

        # 특징 저장
        mel_save = val.replace('.wav', '.ls')
        mfc_save = val.replace('.wav', '.mfc')
        
        np.savetxt(f"{DATA_PATH}/mel/fold{fileFold[idx]}/{mel_save}", mel_feat)
        np.savetxt(f"{DATA_PATH}/mfc/fold{fileFold[idx]}/{mfc_save}", mfc_feat)
