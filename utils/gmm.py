import os
import joblib
import random
import numpy as np

from itertools import chain

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import completeness_score

from config.config import get_config

config = get_config()


class GMM():
    def __init__(self):
        self.n_fold = config.n_fold
        self.data_path = config.load_path
        self.model_path = os.path.join(config.save_path, config.data, config.model)
        self.data_type = config.data
        self.isPCA = config.isPCA
        self.component = config.g_component
        self.n_class = config.n_class

    def fix_frame(self, batch_size, data, fold, label):
        data_shuffle, labels_shuffle = [], []
        data_range = list(range(batch_size))
        random.shuffle(data_range)

        for s in data_range:
            filename = data[s].replace('.wav', '.ls') if self.data_type == 'mel' else data[s].replace('.wav', '.mfc')
            load_path = f"{self.data_path}/{self.data_type}/fold{fold[s]}/{filename}"
            load_feat = np.loadtxt(load_path)

            frame_len = load_feat.shape[-1]
            if frame_len < config.n_frame: continue

            sub_frames = np.array_split(load_feat, config.n_frame, axis=1)
            sub_frames = np.array([np.mean(x, axis=1) for x in sub_frames]).transpose()

            data_shuffle.append(sub_frames.flatten())
            labels_shuffle.append(label[s])

        labels_shuffle = np.eye(self.n_class)[labels_shuffle]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def train(self, k, train_set, valid_set):
        train_mfccs, train_folds, train_labels = zip(*list(chain(*train_set)))
        train_mfccs, train_folds, train_labels = np.array(train_mfccs), np.array(train_folds), np.array(train_labels)

        train_sample = len(train_mfccs)
        train_x, _ = self.fix_frame(train_sample, train_mfccs, train_folds, train_labels)

        # Test Model
        valid_mfccs, valid_folds, valid_labels = zip(*valid_set)
        valid_mfccs, valid_folds, valid_labels = np.array(valid_mfccs), np.array(valid_folds), np.array(valid_labels)

        valid_sample = len(valid_mfccs)
        valid_x, valid_y = self.fix_frame(valid_sample, valid_mfccs, valid_folds, valid_labels)

        if config.isPCA:
            pca = PCA(n_components=config.n_pca)
            pca.fit(train_x)
            train_x = pca.transform(train_x)
            valid_x = pca.transform(valid_x)

        gmm = GaussianMixture(n_components=self.component).fit(train_x)
        joblib.dump(gmm, f"{self.model_path}/gmm-{k}.pkl")

        score = completeness_score(gmm.predict(valid_x), np.argmax(valid_y, axis=1))
        print('Accuracy:{0:.3f}'.format(score))

    def test(self, k, test_set):
        # Test Model
        load_gmm = joblib.load(f"{self.model_path}/gmm-{k}.pkl")
        test_mfccs, test_folds, test_labels = zip(*test_set)
        test_mfccs, test_folds, test_labels = np.array(test_mfccs), np.array(test_folds), np.array(test_labels)

        test_samples = len(test_mfccs)
        test_x, test_y = self.fix_frame(test_samples, test_mfccs, test_folds, test_labels)

        score = completeness_score(load_gmm.predict(test_x), np.argmax(test_y, axis=1))
        return score

