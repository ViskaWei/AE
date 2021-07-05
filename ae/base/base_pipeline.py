import os
import sys
import json
import logging
import argparse
import numpy as np
from dotmap import DotMap


class BasePipeline(object): 
    def __init__(self):
        self.parser = None
        self.config = None
        self.args = None

        ######################### Init ########################
        
    def add_args(self, parser):
        parser.add_argument(
                '--config',
                dest='config',
                metavar='C',
                default=None,
                help='The Configuration file')
        # parser.add_argument('--config', type=str, nargs='+', help='Load config from json file.')
        
    def create_parser(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.add_subparsers(self.parser)

    def add_subparsers(self, parser):
        self.add_args(parser)
    
    def parse_args(self):
        if self.args is None:
            args_dict = self.parser.parse_args().__dict__
            self.args = DotMap(args_dict)
            if "config" in self.args:
                self.load_config_from_json(self.args.config)
            else:
                raise NotImplementedError

    def load_config_from_json(self, filename):
        """
        Get the config from a json file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
        # parse the configurations from the config json file provided
        with open(filename, 'r') as f:
            config_dict = json.load(f)

        # convert the dictionary to a namespace using bunch lib
        self.config = DotMap(config_dict)

###########################################Arg Parse Util Fn ###############################################
    def is_arg(self, name, args=None):
        args = args or self.args
        return name in args and args[name] is not None

    def get_arg(self, name, args=None, default=None):
        args = args or self.args
        if name in args and args[name] is not None: 
            return args[name]
        elif default is not None:
            return default
        else:
            raise Exception(f'arg {name} not specified in command line')

    def update_config(self, config_class, name):
        if name in self.args and self.args[name] is not None: 
            self.config[config_class][name] =  self.args[name]
        # else:
        #     raise Exception(f'arg {name} not specified in command line')
    
    def get_loop_from_arg(self, name, fn = None):
        loopArgs = self.get_arg(name)
        if fn is None:
            loopFn = lambda x: int(loopArgs[0]**(loopArgs[1] + x))
        else:
            loopFn = fn
        return [loopFn(ii) for ii in range(loopArgs[2])]


############################## APPLY ARGS ###########################
        
    def apply_args(self):
        pass
        # self.apply_init_args()
        # self.apply_debug_args()
        # self.apply_input_args()
        # self.apply_output_args()
    
    def apply_init_args(self):
        if self.is_arg('seed'): np.random.seed(self.args['seed'])
        self.device = self.get_arg('device', default='cpu') 

    def apply_input_args(self):
        self.inDir = self.get_arg('in') 
        # self.dim = self.get_arg('dim')
  
    def apply_output_args(self):       
        out = self.get_arg('out')
        self.outDir = out
        self.logDir = './log/'
        self.create_output_dir(self.outDir)
        # self.create_output_dir(self.logDir)

    def apply_debug_args(self):
        self.debug = self.get_arg('debug')


    def create_output_dir(self, dir, cont=False):
        try: 
            os.mkdir(dir)
            # logging.info('Creating directory {}'.format(dir))
        except:
            pass
            logging.info('Output directory not Empty, Replacing might occurs')
        # os.path.exists(dir):
        #     if len(os.listdir(dir)) != 0:
        #         print('Output directory not Empty, Replacing might occurs')

    # def init_logging(self, outdir):
    #     self.setup_logging(os.path.join(outdir, type(self).__name__.lower() + '.log'))
    #     # handler = logging.StreamHandler(sys.stdout)
    #     # handler.setLevel(logging.DEBUG)
    #     # handler.setFormatter(formatter)
    #     # root.addHandler(handler)
    
    def setup_logging(self, logfile=None):
        logging.basicConfig()
        # logging.basicConfig(filename=f'.log', encoding='utf-8', level=logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(self.get_logging_level())
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def get_logging_level(self):
        # return logging.DEBUG if self.debug else logging.INFO
        return logging.INFO

    def prepare(self):
        self.create_parser()
        self.parse_args() 
        self.setup_logging()
        self.apply_args()
    
    def run(self):
        pass

    def finish(self):
        pass

    def execute(self):
        logging.info("================================LETS GO=================================")

        self.prepare()
        self.run()
        self.finish()
