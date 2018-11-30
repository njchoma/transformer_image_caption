import os
import yaml
import logging
import pickle
import argparse
import matplotlib
import time
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
def train_one_epoch(args, model, train_loader, optimizer, len_vocab):
    model.train()
    nb_batch = len(train_loader)
    nb_train = nb_batch * args.batch_size
    logging.info("Training {} batches, {} samples.".format(nb_batch, nb_train))
    
    loss = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0
    for i, (features, captions, lengths) in enumerate(train_loader):
        #print(i)
        len_captions = len(captions[0])
        if torch.cuda.is_available():
            features, captions = features.cuda(), captions.cuda()

        t0 = time.time()
        print(model(features, 10, 4))
        print("Time: {:2.2f}s".format(time.time() - t0))
        exit()
        out = model(features, len_captions)
        #print(out)
        n_ex, vocab_len = out.view(-1, len_vocab).shape
        #print(n_ex)
        #print(vocab_len)
        captions = captions[:,1:]
        #print(captions.contiguous().view(1,n_ex).squeeze().shape)
        #nonzeros = torch.nonzero(captions[:,1:,:])        
        #print(nonzeros.shape)
        #caption_indices = captions[:,1:,:] == 1
        
        batch_loss = loss(out.view(-1,len_vocab),captions.contiguous().view(1, n_ex).squeeze())
        #print(str(batch_loss.item()) + "\n")
        epoch_loss+=batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()        

    epoch_loss = epoch_loss/nb_batch 
    logging.info("Train loss: " + str(epoch_loss))
    return epoch_loss

def val_one_epoch(args, model, val_loader, optimizer, len_vocab, beam=None):
    model.eval()
    nb_batch = len(val_loader)
    nb_val = nb_batch * args.batch_size
    logging.info("Validating {} batches, {} samples.".format(nb_batch, nb_val))

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0

    with torch.no_grad():
        for i, (features, captions, lengths) in enumerate(val_loader):
        #print(i)
            len_captions = len(captions[0])
            if torch.cuda.is_available():
                features, captions = features.cuda(), captions.cuda()

            if beam is not None:
                if (i % 200) == 0:
                    sentences = model(features, 20, beam)
                    print(sentences)
            out = model(features, len_captions)
            n_ex, vocab_len = out.view(-1, len_vocab).shape
            captions = captions[:,1:]
            batch_loss = loss(out.view(-1,len_vocab),captions.contiguous().view(1, n_ex).squeeze())
            epoch_loss+=batch_loss.item()
   
    epoch_loss = epoch_loss/nb_batch
    logging.info("Val loss: " + str(epoch_loss))
    return epoch_loss

def train(args, model, train_loader, val_loader, len_vocab):
    logging.warning("Beginning training")
    optimizer = None
    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.899999976158, weight_decay=0.000500000023749)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if torch.cuda.is_available():
        model = model.cuda()

    train_loss_array = []
    val_loss_array = []

    train_epoch_array = []
    val_epoch_array = []

    '''
    val_loss = val_one_epoch(args,model,val_loader, optimizer, len_vocab)
    logging.info("Validation loss with random initialization: " + str(val_loss))
    '''
    
    logging.info("No of epochs: " + str(args.max_nb_epochs))
    while args.current_epoch < args.max_nb_epochs:
        args.current_epoch += 1
        logging.info("\nEpoch {}".format(args.current_epoch))
        
        t0=time.time()
        train_loss = train_one_epoch(args, model, train_loader, optimizer, len_vocab)
        logging.info("Train done in: {:3.1f} seconds".format(time.time() - t0))
        
        t0 = time.time()
        val_loss = val_one_epoch(args,model,val_loader, optimizer, len_vocab, beam=3)
    
        logging.info("Valid done in: {:3.1f} seconds".format(time.time() - t0))

        torch.save({
            'epoch': args.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.experiment_dir, "epoch_" + str(args.current_epoch) + ".pth.tar"))

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

        train_epoch_array.append(args.current_epoch)
        val_epoch_array.append(args.current_epoch)

    plt.plot(train_epoch_array, train_loss_array, label='Training loss')
    plt.plot(val_epoch_array, val_loss_array, label = 'Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.experiment_dir, "loss_stats.png"))
    

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

    val_loader = get_loader(data_root=args.root_dir,
                              vocab=vocab,
                              batch_size=args.batch_size,
                              data_type='val',
                              shuffle=True,
                              num_workers=0,
                              debug=args.debug)
    test_loader  = None
    logging.info("Done.")

    model = Caption_Model(dict_size=len(vocab),
                          image_feature_dim=feature_dim, vocab=vocab)
    logging.info(model)
    train(args, model, train_loader, val_loader, len(vocab))

if __name__ == "__main__":
    main()
