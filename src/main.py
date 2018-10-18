import os
import yaml
import logging
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.caption_model import Caption_Model
from data import initialize_data

#########################################
#               CONSTANTS               #
#########################################
ARGS_FILE = 'args.yml'

#############################
#           DEVICE          #
#############################
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#####################################
#               UTILS               #
#####################################
def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--train_dir', type=str, required=True,
            help='Folder where training data is stored')
    add_arg('--artifacts_dir', type=str, required=True,
            help='Folder where all training artifacts are saved')
    add_arg('--name', type=str, required=True,
            help='Name of current training scheme')
    add_arg('--run_nb', type=int, default=0,
            help='Run number of current iteration of model')

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

#########################################
#               TRAINING                #
#########################################
def train(args, model, train_loader, valid_loader):
    logging.warning("Beginning training")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

#####################################
#               MAIN                #
#####################################
def main():
    args = read_args()
    args.experiment_dir = os.path.join(args.artifacts_dir,
                                       args.name,
                                       str(args.run_nb))
    os.makedirs(args.experiment_dir, exist_ok=True)
    initialize_logger(args.experiment_dir)
    try:
        load_args(args.experiment_dir, ARGS_FILE)
    except:
        save_args(args, args.experiment_dir, ARGS_FILE)

    logging.warning("Starting {}, run {}.".format(args.name, args.run_nb))

    # TODO construct train, valid dataloader
    logging.info("Loading data...")
    dictionary, image_feature_dim = initialize_data()
    train_loader = None
    valid_loader = None
    test_loader  = None
    logging.info("Done.")

    model = Caption_Model(dict_size=10, image_feature_dim=image_feature_dim)

    train(args, model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
