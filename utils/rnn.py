import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from itertools import chain
from sklearn.decomposition import PCA

from config.config import get_config

config = get_config()

class LSTM():
    def __init__(self):
        self.n_fold = config.n_fold
        self.data_path = config.load_path
        self.model_path = os.path.join(config.save_path, config.data, config.model)
        self.data_type = config.data
        self.n_class = config.n_class
        self.step = config.train_step
        self.n_unit_1 = config.n_unit_1
        self.n_unit_2 = config.n_unit_2
        self.n_unit_3 = config.n_unit_3
        self.learning_rate = config.learning_rate
        self.dimension = config.mel if self.data_type == 'mel' else config.mfc*3

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

            data_shuffle.append(sub_frames)
            labels_shuffle.append(label[s])

        labels_shuffle = np.eye(self.n_class)[labels_shuffle]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def neural_net(self):
        inputs = keras.Input([self.dimension, config.n_frame])
        lstm_1 = keras.layers.LSTM(units=self.n_unit_1, activation=tf.nn.sigmoid, return_sequences=True)(inputs)
        lstm_2 = keras.layers.LSTM(units=self.n_unit_1, activation=tf.nn.sigmoid, return_sequences=True)(lstm_1)
        lstm_3 = keras.layers.LSTM(units=self.n_unit_1, activation=tf.nn.sigmoid)(lstm_2)
        output_layer = keras.layers.Dense(self.n_class, activation=tf.nn.softmax)(lstm_3)
        return keras.Model(inputs=inputs, outputs=output_layer)

    def train(self, k, train_set, valid_set):
        train_wavs, train_folds, train_labels = zip(*list(chain(*train_set)))
        train_wavs, train_folds, train_labels = np.array(train_wavs), np.array(train_folds), np.array(train_labels)

        train_sample = len(train_wavs)
        train_x, train_y = self.fix_frame(train_sample, train_wavs, train_folds, train_labels)

        valid_wavs, valid_folds, valid_labels = zip(*valid_set)
        valid_wavs, valid_folds, valid_labels = np.array(valid_wavs), np.array(valid_folds), np.array(valid_labels)

        valid_sample = len(valid_wavs)
        valid_x, valid_y = self.fix_frame(valid_sample, valid_wavs, valid_folds, valid_labels)

        if config.isPCA:
            pca = PCA(n_components=config.n_pca)
            pca.fit(train_x)
            train_x = pca.transform(train_x)
            valid_x = pca.transform(valid_x)

        model = self.neural_net()
        # model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        for step in range(1, self.step + 1):
            model.fit(train_x, train_y, epochs=10, verbose=1, validation_data=(valid_x, valid_y))  # train
            _, score = model.evaluate(valid_x, valid_y, verbose=0)
            print('Accuracy:{0:.3f}'.format(score))

        model.save(f"{self.model_path}/rnn-{k}.h5")

    def test(self, k, test_set):
        load_dnn = tf.keras.models.load_model(f"{self.model_path}/rnn-{k}.h5")
        test_wavs, test_folds, test_labels = zip(*test_set)
        test_wavs, test_folds, test_labels = np.array(test_wavs), np.array(test_folds), np.array(test_labels)

        test_samples = len(test_wavs)
        test_x, test_y = self.fix_frame(test_samples, test_wavs, test_folds, test_labels)

        _, score = load_dnn.evaluate(test_x, test_y, verbose=0)
        return score