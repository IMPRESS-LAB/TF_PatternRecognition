import os
from config.config import get_config

config = get_config()

if __name__ == '__main__':
    step = config.step

    if step < 0 or step > 2:
        raise ValueError("Please enter only between 0 and 2.")

    # Feature extracting step
    if step <= 0:
        print('Extracting features...')
        os.system('python ./utils/feature_extractor.py')
        print('Process done!')

    # Model training step
    if step <= 1:
        print('Training model...')
        os.system(f'python train.py --isTrain 1 --data {config.data} --model {config.model}')
        print('Process done!')

    # Model inference step
    if step <= 2:
        print('Testing model...')
        os.system(f'python train.py --isTrain 0 --data {config.data} --model {config.model}')
        print('Process done!')