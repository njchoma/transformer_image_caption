import os
import yaml
import logging
import pickle
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils_experiment as utils
from data_helpers.data_loader import get_loader
from data_helpers.vocab import Vocabulary
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
    feature_dim = 2048
    nb_features = 36
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
    with open(os.path.join(args.root_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    train_loader = get_loader(data_root=args.root_dir,
                              vocab=vocab,
                              batch_size=args.batch_size,
                              data_type='train',
                              shuffle=True,
                              num_workers=0)
    valid_loader = None
    test_loader  = None
    logging.info("Done.")

    model = Caption_Model(dict_size=len(vocab),
                          image_feature_dim=feature_dim)

    train(args, model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
