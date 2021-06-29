import os
import h5py
import numpy as np
# import timeit
import logging
from tqdm import tqdm

from ae.data_loader.spec_data_loader import SpecDataLoader
from ae.model.simple_ae_model import SimpleAEModel
from ae.trainer.simple_ae_trainer import SimpleAETrainer

from ae.util.args import get_args
from ae.util.config import process_config


# DATA_DIR = '/scratch/ceph/swei20/data/ae/dataset/'
# AE_DIR = os.path.join(DATA_DIR, 'ae')
# BASE_DIR = '/home/swei20/AE'
# OUT_DIR = os.path.join(BASE_DIR, 'ae')

# logging.basicConfig(level=logging.INFO)
# logging.info(f'RUNNING RD {RD}')


CONFIG_PATH = '/home/swei20/AE/configs/ae/train/config.json'

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args(default=CONFIG_PATH)
    config = process_config(args.config)

    # create the experiments dirs
    # create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = SpecDataLoader(config)
    data_loader.get_train_data()


    print('Create the model.')
    model = SimpleAEModel()
    model.model
    # print('Create the trainer')
    # trainer = SimpleMnistModelTrainer(model.model, data_loader.get_train_data(), config)

    # print('Start training the model.')
    # trainer.train()


if __name__ == '__main__':
    main()
