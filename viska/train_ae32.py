import os
import h5py
import numpy as np
# import timeit
import logging
from tqdm import tqdm

DATA_DIR = '/scratch/ceph/swei20/data/pfsspec'
AE_DIR = os.path.join(DATA_DIR, 'ae')
BASE_DIR = '/home/swei20/project/pfs_spec_dnn/viska/'
# OUT_DIR = os.path.join(BASE_DIR, 'ae')

logging.basicConfig(level=logging.INFO)
logging.info(f'RUNNING RD {RD}')

