import os
import h5py
import json
import numpy as np
# import timeit
import logging
from tqdm import tqdm

from ae.base.base_pipeline import BasePipeline
# from ae.data_loader.spec_data_loader import SpecDataLoader
from ae.data_loader.pc_data_loader import PcDataLoader
from ae.model.simple_ae_model import SimpleAEModel
from ae.model.vae_model import VAEModel

from ae.trainer.simple_ae_trainer import SimpleAETrainer

from ae.util.args import get_args
from ae.util.config import process_config


class AEPipelineTrace():
    def __init__(self):
        self.dfEval = None

class AEPipeline(BasePipeline):
    def __init__(self, logging=True, trace=None):
        super().__init__()

        self.trace = trace
        # self.model = None

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--type', type=str, choices=["ae", "vae"], help='Choose from ae or vae\n' )
        parser.add_argument("--stddev", type=float, default=None)
        

        parser.add_argument('--lr', type=float, default=None, help='Learning Rate\n' )
        parser.add_argument('--dropout', type=float, default=None, help='Dropout Rate\n' )
        # parser.add_argument('--std-rate', type=float, default=None, help='std scaled Rate\n' )
        

        parser.add_argument('--epoch', type=int, default=None, help='Num of Epochs\n' )
        parser.add_argument('--verbose', type=int, default=None, help='Verbose Training\n' )
        parser.add_argument('--hidden-dims', 
                            type=lambda s: [int(item) for item in s.split(',')], 
                            nargs='+', 
                            default=None, help='Hidden layers\n' )
        parser.add_argument('--save', type=int, default=None, help='saving model\n' )


    def prepare(self):
        super().prepare()
        self.apply_data_args()
        self.apply_model_args()
        self.apply_trainer_args()
        # self.apply_norm_args()
        # self.apply_name_args()

    def apply_data_args(self):
        # self.update_config("data", "std_rate")
        logging.info(self.config.data)


    def apply_model_args(self):
        self.update_config("model", "type")
        self.update_config("model", "lr")
        self.update_config("model", "dropout")
        if self.args["hidden_dims"] is not None:
            if not isinstance(self.args["hidden_dims"][0], int):
                self.args["hidden_dims"] = self.args["hidden_dims"][0]
            self.update_config("model", "hidden_dims")
        if self.config.model.type == "vae":
            self.update_config("model", "stddev")
        logging.info(self.config.model)


    def apply_trainer_args(self):
        self.update_config("trainer", "epoch")
        self.update_config("trainer", "verbose")
        self.update_config("trainer", "save")

        logging.info(self.config.trainer)

        
    def run(self):
        super().run()
        data = self.run_step_data_loader()
        self.run_step_model(data)
        
    def run_step_data_loader(self, config=None):
        config = config or self.config
        # ds = SpecDataLoader()
        ds = PcDataLoader()
        ds.init_from_config(config)
        data = ds.get_train_data()
        logging.info(f"train data size: {data[0].shape}")
        return data


    def run_step_model(self, data, config=None):
        config = config or self.config
        mm = self.get_model_type(config.model.type)
        mm.build_model(config)
        # logging.info(mm.model.summary())
        # logging.info(mm.encoder.summary())
        # logging.info(mm.decoder.summary())


        logging.info(f'Loss: {mm.model.loss}')
        tt = SimpleAETrainer(mm, config)
        history = tt.train(data)
        if not config.trainer.verbose:
            ep = len(mm.model.history.history["lr"])
            prints =f"| EP {ep} |"
            for key, value in mm.model.history.history.items():
                prints = prints +  f"{key}: {np.around(value[-1],3)} | "
            logging.info(prints)
            # acc, val_acc = self.get_last_epoch_accs(mm.model)
            # logging.info(f"ACC {acc}% | VACC {val_acc}%")
        if config.trainer.save: 
            mm.save()

    # def run_step_save(self, model):
    #     self.model.save()

    def get_model_type(self, type):
        if type == "ae":
            logging.info("Using AE Model")
            return SimpleAEModel()
        elif type == "vae":
            logging.info("Using VAE Model")
            return VAEModel()

    def get_last_epoch_accs(self, model):
        val_acc = model.history.history['val_acc'][-1]
        acc = model.history.history['acc'][-1]
        return np.around(acc*100, 2), np.around(val_acc*100, 2)


    def finish(self):
        super().finish()
        logging.info(self.config.data)
        logging.info(self.config.model)
        logging.info("=============================================================")
