import os
import yaml
import logging
import argparse

#####################################
#               UTILS               #
#####################################
def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--root_dir', type=str, required=True,
            help='Folder where train, val, test data is stored')
    add_arg('--artifacts_dir', type=str, required=True,
            help='Folder where all training artifacts are saved')
    add_arg('--name', type=str, required=True,
            help='Name of current training scheme')
    add_arg('--run_nb', type=int, default=0,
            help='Run number of current iteration of model')
    add_arg('--batch_size', type=int, default=1,
            help='Minibatch size')

    return parser.parse_args()

def initialize_logger(experiment_dir):
    logfile = os.path.join(experiment_dir, 'log.txt')
    logging.basicConfig(filename=logfile,format='%(message)s',level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

def save_args(args, experiment_dir, args_file):
    args_path = os.path.join(experiment_dir, args_file)
    with open(args_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False)
    logging.info("Args saved")
    
def load_args(experiment_dir, args_file):
    args_path = os.path.join(experiment_dir, args_file)
    with open(args_path, 'r') as f:
        args = yaml.load(f)
    logging.info("Args loaded")
    return args
