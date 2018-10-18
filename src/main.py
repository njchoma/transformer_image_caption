import os
import logging
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.caption_model import Caption_Model
from data import initialize_data

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
    logging.warning("Starting {}, run {}.".format(args.name, args.run_nb))

    # TODO construct train, valid dataloader
    logging.info("Loading data...")
    dictionary = initialize_data()
    train_loader = None
    valid_loader = None
    test_loader  = None
    logging.info("Done.")

    model = Caption_Model(dict_size=10, image_feature_dim=100)

    train(args, model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
