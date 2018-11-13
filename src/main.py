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
def train_one_epoch(args, model, train_loader, optimizer):
    nb_batch = len(train_loader)
    nb_train = nb_batch * args.batch_size
    logging.info("Training {} batches, {} samples.".format(nb_batch, nb_train))
    for i, (features, captions, lengths) in enumerate(train_loader):
        print(i)
        out = model(features, len(captions[0]), captions)
        print(out)
        #write loss and backprop
        exit()

def train(args, model, train_loader, valid_loader):
    logging.warning("Beginning training")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    while args.current_epoch < args.max_nb_epochs:
        args.current_epoch += 1
        logging.info("\nEpoch {}".format(args.current_epoch))
        train_stats = train_one_epoch(args, model, train_loader, optimizer)

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
    if os.path.exists(args.experiment_dir)==False:
        os.makedirs(args.experiment_dir)
    utils.initialize_logger(args.experiment_dir)
    logging.warning("Starting {}, run {}.".format(args.name, args.run_nb))
    try:
        args = utils.load_args(args.experiment_dir, ARGS_FILE)
    except:
        args.current_epoch = 0
        utils.save_args(args, args.experiment_dir, ARGS_FILE)


    logging.info("Loading data...")
    with open(os.path.join(args.root_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    train_loader = get_loader(data_root=args.root_dir,
                              vocab=vocab,
                              batch_size=args.batch_size,
                              data_type='train',
                              shuffle=True,
                              num_workers=0,
                              debug=args.debug)
    valid_loader = None
    test_loader  = None
    logging.info("Done.")

    model = Caption_Model(dict_size=len(vocab),
                          image_feature_dim=feature_dim, vocab=vocab)

    train(args, model, train_loader, valid_loader)

if __name__ == "__main__":
    main()
