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
parser.add_argument('--isTrain', type=str2bool, default=True, help='Choose True(1) if you want to train')
parser.add_argument('--isPCA', type=str2bool, default=False, help='Choose True(1) if you want to PCA')
parser.add_argument('--n_pca', type=int, default=200, help='PCA dimension to apply (pca_dim <= original_dim')
parser.add_argument('--n_frame', type=int, default=50, help='Input length to the network for training')
parser.add_argument('--n_class', type=int, default=10, help='Number of classes')

# GMM
parser.add_argument('--g_component', type=int, default=10, help='Number of components in GMM')
# HMM
parser.add_argument('--h_component', type=int, default=10, help='Number of states in HMM')
# DNN, RNN
parser.add_argument('--train_step', type=int, default=2, help='Number of train steps in DNN, CNN and RNN')
parser.add_argument('--n_unit_1', type=int, default=512, help='Number of first layer units in DNN, CNN and RNN')
parser.add_argument('--n_unit_2', type=int, default=256, help='Number of second layer units in DNN, CNN and RNN')
parser.add_argument('--n_unit_3', type=int, default=64, help='Number of third layer units in DNN and RNN')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate in DNN, CNN and RNN')
# CNN
parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size of convolution layers in CNN')
parser.add_argument('--conv_filters', type=int, default=64, help='Filter number of convolution layers in CNN')
parser.add_argument('--pool_size', type=int, default=2, help='Pool size of pooling layers in CNN')
parser.add_argument('--padding', type=str, default='same', help='Type of padding in CNN')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Rate of dropout in CNN')