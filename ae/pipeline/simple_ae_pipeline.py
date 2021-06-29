import os
import h5py
import numpy as np
# import timeit
import logging
from tqdm import tqdm

from ae.base.base_pipeline import BasePipeline
from ae.data_loader.spec_data_loader import SpecDataLoader
from ae.model.simple_ae_model import SimpleAEModel
from ae.trainer.simple_ae_trainer import SimpleAETrainer

from ae.util.args import get_args
from ae.util.config import process_config


class SimpleAEPipelineTrace():
    def __init__(self):
        self.dfEval = None

class SimpleAEPipeline(BasePipeline):
    def __init__(self, logging=True, trace=None):
        super().__init__()

        self.trace = trace

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--lr', type=float, default=None, help='Learning Rate\n' )
        parser.add_argument('--epoch', type=int, default=None, help='Num of Epochs\n' )

    def prepare(self):
        super().prepare()
        self.apply_model_args()
        self.apply_trainer_args()
        # self.apply_norm_args()
        # self.apply_name_args()

    def apply_model_args(self):
        self.update_config("model", "lr")

    def apply_trainer_args(self):
        self.update_config("trainer", "epoch")
        
    def run(self, config=None):
        config = config or self.config
        data = self.run_step_data_loader(config)
        self.run_step_model(config, data)
        
    def run_step_data_loader(self, config):
        ds = SpecDataLoader()
        ds.init_from_config(config)
        data = ds.get_train_data()
        logging.info(f"train data size: {data[0].shape}")
        return data

    def run_step_model(self, config, data):
        mm = SimpleAEModel()
        mm.build_model(config)
        tt = SimpleAETrainer(mm, config)
        history = tt.train(data)