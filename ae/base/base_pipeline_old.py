import os
import sys
import json
import logging
import argparse
import numpy as np

class BasePipeline(object): 
    def __init__(self):
        self.parser = None
        self.args = None
        self.inDir = None 
        self.out = None
        self.outDir= None
        self.logDir= None

        self.dim=None
        self.debug=False
        self.device=None
        self.test = False
        ######################### Init ########################
        
    def add_args(self, parser):
        parser.add_argument('--config', type=str, nargs='+', help='Load config from json file.')
        parser.add_argument('--seed', type=int, help='Set random\n' )
        parser.add_argument('--in', type=str, help='input dir\n')
        parser.add_argument('--out', type=str, help='output dir\n')
        parser.add_argument('--name', type=str, help='save model name\n')
        parser.add_argument('--device', type=str, help='device cpu or cuda\n')
        parser.add_argument('--test', action = 'store_true', help='Test or original size\n')

        # parser.add_argument('--dim', type=int, default=None,  help='Latent Representation dimension\n')
        parser.add_argument('--debug', action ='store_true', help='debug mode\n')
        
    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.add_subparsers(self.parser)

    def add_subparsers(self, parser):
        self.add_args(parser)
    
    def parse_args(self):
        if self.args is None:
            self.args = self.parser.parse_args().__dict__
            self.get_configs(self.args)
        # print(self.args)
    
    def is_arg(self, name, args =None):
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
            
    
    def get_loop_from_arg(self, name, fn = None):
        loopArgs = self.get_arg(name)
        if fn is None:
            loopFn = lambda x: int(loopArgs[0]**(loopArgs[1] + x))
        else:
            loopFn = fn
        return [loopFn(ii) for ii in range(loopArgs[2])]

    def get_config_args(self, args=None):
        args = args or self.args
        for key, val in args.items():
            if self.is_arg(key,args=args):
                self.args[key] = val
    
    def update_nested_configs(self,args):
        args = self.load_args_jsons(args)
        while self.is_arg('config',args=args): 
            configArg = self.load_args_jsons(args)
            del args['config']
            args.update(configArg)
        return args

    def get_configs(self, args):
        args = self.update_nested_configs(args)
        # print('configArgs', args)
        self.get_config_args(args)
        # print('get_configs_final',self.args)
    
    def load_args_jsons(self, args):
        configFiles = self.get_arg('config', args=args)
        for configFile in configFiles:
            configArg = self.load_args_json(configFile)
            args.update(configArg)
        return args

    def load_args_json(self, filename):
        # print(filename)
        with open(filename, 'r') as f:
            args = json.load(f)
        return args
    
############################## APPLY ARGS ###########################
        
    def apply_args(self):
        self.apply_init_args()
        self.apply_debug_args()
        self.apply_input_args()
        self.apply_output_args()
    
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
        return logging.DEBUG if self.debug else logging.INFO
            
    def prepare(self):
        self.create_parser()
        self.parse_args() 
        self.setup_logging()
        self.apply_args()
    
    def run(self):
        pass

