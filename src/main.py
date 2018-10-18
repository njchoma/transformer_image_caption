import os
import yaml
import logging
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils_experiment as utils
from data import initialize_data
from models.caption_model import Caption_Model

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
    args = utils.read_args()
    args.experiment_dir = os.path.join(args.artifacts_dir,
                                       args.name,
                                       str(args.run_nb))
    os.makedirs(args.experiment_dir, exist_ok=True)
    utils.initialize_logger(args.experiment_dir)
    try:
        utils.load_args(args.experiment_dir, ARGS_FILE)
    except:
        utils.save_args(args, args.experiment_dir, ARGS_FILE)

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
