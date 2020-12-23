import os
import numpy as np

from config.config import get_config

from utils.feature_extractor import parse_metadata
from utils.kmeans import KMEANS
from utils.gmm import GMM
from utils.hmm import HMM

config = get_config()


if __name__ == '__main__':
    # Read data
    DATA_PATH = config.load_path
    MODEL_TYPE = config.model
    metadata = parse_metadata(DATA_PATH + "/UrbanSound8K.csv")

    if MODEL_TYPE == 'kmeans':
        model = KMEANS()
    elif MODEL_TYPE == 'gmm':
        model = GMM()
    elif MODEL_TYPE == 'hmm':
        model = HMM()
    else:
        raise ValueError
    os.makedirs(model.model_path, exist_ok=True)

    if config.isTrain:    # Train
        print('==================================================')
        print('Training Start!')
        for fold in range(config.n_fold):
            print('# %d-Fold' % (fold + 1), end='\t')

            # Train/Test(Validation) split
            select_valid = metadata.pop(0)
            select_train = metadata.copy()
            metadata.insert(10, select_valid)

            model.train(fold, select_train, select_valid)
        print('Training Finish!')
        print()

    else:
        Acc = []
        for fold in range(config.n_fold):
            # Train/Test(Validation) split
            select_test = metadata.pop(0)
            select_train = metadata.copy()
            metadata.insert(10, select_test)

            Acc.append(model.test(fold, select_test))

        print('{0:s}\tAccuracy:{1:.3f}'.format(config.model.upper(), np.mean(Acc)))