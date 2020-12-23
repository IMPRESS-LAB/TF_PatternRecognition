import argparse

parser = argparse.ArgumentParser()


def get_config():
    config, unparsed = parser.parse_known_args()
    return config


def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--step', type=int, default=2, help='Running step')
# PATH
parser.add_argument('--load_path', type=str, default='./data', help='Path to the dataset')
parser.add_argument('--save_path', type=str, default='./model', help='Path for model')
parser.add_argument('--n_fold', type=int, default=10, help='Number of fold')

# AUDIO
parser.add_argument('--data', type=str, default='mel', help='Choose type of data (mel/mfc)')
parser.add_argument('--n_fft', type=int, default=512, help='Number of FFT components')
parser.add_argument('--sr', type=int, default=8000, help='Sampling rate of incoming signal')
parser.add_argument('--win_len', type=int, default=100, help='Number of samples between successive frames')
parser.add_argument('--hop_len', type=int, default=200, help='Each frame of audio is windowed by window()')
parser.add_argument('--mel', type=int, default=40, help='Feature dimension of Mel-spectrum')
parser.add_argument('--mfc', type=int, default=13, help='Feature dimension of MFCC')
parser.add_argument('--filter', type=int, default=23, help='Number of Mel bands to generate for MFCC')

# MODEL
parser.add_argument('--model', type=str, default='gmm', help='Choose type of model (kmeans/gmm/hmm/dnn/cnn/rnn')
parser.add_argument('--isTrain', type=str2bool, required=True, help='Choose True(1) if you want to train')
parser.add_argument('--isPCA', type=str2bool, default=False, help='Choose True(1) if you want to PCA')
parser.add_argument('--n_pca', type=int, default=200, help='PCA dimension to apply (pca_dim <= original_dim')
parser.add_argument('--n_frame', type=int, default=50, help='Input length to the network for training')
parser.add_argument('--n_class', type=int, default=10, help='Number of classes')

# GMM
parser.add_argument('--g_component', type=int, default=128, help='Number of components in GMM')
# HMM
parser.add_argument('--h_component', type=int, default=100, help='Number of components in HMM')
# DNN
# CNN
# RNN (LSTM)
