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
from data_helpers.data_loader_ks import get_loader
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
    for i, (image_ids, features, captions, lengths) in enumerate(train_loader):
        #print(i)
        len_captions = len(captions[0])
        if torch.cuda.is_available():
            features, captions = features.cuda(), captions.cuda()

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
    logging.info("Train loss: {:>.3E}".format(epoch_loss))
    return epoch_loss

def val_one_epoch(args, model, val_loader, optimizer, len_vocab, beam=None):
    model.eval()
    nb_batch = len(val_loader)
    nb_val = nb_batch * args.batch_size
    logging.info("Validating {} batches, {} samples.".format(nb_batch, nb_val))

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0

    with torch.no_grad():
        for i, (image_ids, features, captions, lengths) in enumerate(val_loader):
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
    logging.info("Val loss: {:>.3E}".format(epoch_loss))
    return epoch_loss

def test(args, model, test_loader, len_vocab, beam=None):
    model.eval()
    nb_batch = len(test_loader)
    nb_test = nb_batch * args.batch_size
    logging.info("Testing {} batches, {} samples.".format(nb_batch, nb_test))

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0

    with torch.no_grad():
        for i, (image_ids, features, captions, lengths) in enumerate(test_loader):
        #print(i)
            len_captions = len(captions[0])
            if torch.cuda.is_available():
                features, captions = features.cuda(), captions.cuda()

            if beam is not None:
                if (i % (nb_batch//2)) == 0:
                    sentences = model(features[0:1], 20, beam)
                    print(sentences)
            out = model(features, len_captions)
            n_ex, vocab_len = out.view(-1, len_vocab).shape
            captions = captions[:,1:]
            batch_loss = loss(out.view(-1,len_vocab),captions.contiguous().view(1, n_ex).squeeze())
            epoch_loss+=batch_loss.item()
   
    epoch_loss = epoch_loss/nb_batch
    logging.info("Test loss: " + str(epoch_loss))
    return epoch_loss

def save_final_captions(args, model, val_loader, max_sent_len, beam_width):
    nb_batch = len(val_loader)
    nb_val = nb_batch * args.batch_size
    assert nb_batch == nb_val # Must be equal for beam search
    logging.info("Captioning {} batches, {} samples.".format(nb_batch, nb_val))

    s = utils.Sentences(args.experiment_dir)
    model.eval()
    with torch.no_grad():
        for i, (image_ids, features, captions, lengths) in enumerate(val_loader):
            if torch.cuda.is_available():
                features = features.cuda()
            sentence = model(features, max_sent_len, beam_width)
            s.add_sentence(image_ids[0], sentence[1])
            if (i % 50) == 0:
                logging.info("  {:4d}".format(i))
    logging.info("Saving sentences...")
    s.save_sentences()
    logging.info("Done.")


def train(args, model, train_loader, val_loader, len_vocab):
    logging.warning("Beginning training")
    optimizer = None


    if args.current_epoch == 0:
        val_loss = val_one_epoch(args,model,val_loader, optimizer, len_vocab, beam=None)
        logging.info("Validation loss with random initialization: " + str(val_loss))
    
    logging.info("Maximum of epochs: " + str(args.max_nb_epochs))
        

    while args.current_epoch < args.max_nb_epochs:
        args.current_epoch += 1
        logging.info("\nEpoch {}".format(args.current_epoch))
        
        t0=time.time()
        train_loss = train_one_epoch(args, model, train_loader, optimizer, len_vocab)
        logging.info("Train done in: {:3.1f} seconds".format(time.time() - t0))
        
        t0 = time.time()
        val_loss = val_one_epoch(args,model,val_loader, optimizer, len_vocab, beam=None)
    
        logging.info("Valid done in: {:3.1f} seconds".format(time.time() - t0))

        torch.save({
            'epoch': args.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(args.experiment_dir, "epoch_" + str(args.current_epoch) + ".pth.tar"))

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

        train_epoch_array.append(args.current_epoch)
        val_epoch_array.append(args.current_epoch)

        #keep a track of the best model and save it
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save({
                'epoch': args.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(args.experiment_dir, "best_model"  + ".pth.tar"))
    
    plt.plot(train_epoch_array, train_loss_array, label='Training loss')
    plt.plot(val_epoch_array, val_loss_array, label = 'Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.experiment_dir, "loss_stats.png"))

def create_model(args, vocab, feature_dim):
    model = None
    
    model = Caption_Model(dict_size=len(vocab),
                            image_feature_dim=feature_dim, vocab=vocab)
    
    if args.resume_epoch > 0:
        logging.info('Loading checkpoint')
        args.checkpoint = torch.load(os.path.join(args.experiment_dir, "epoch_" + str(args.resume_epoch) + ".pth.tar" ))
        model.load_state_dict(args.checkpoint['model_state_dict'])
        args.current_epoch = args.resume_epoch
    elif args.resume_epoch < 0:
        logging.info('Loading best checkpoint')
        args.checkpoint = torch.load(os.path.join(args.experiment_dir, "best_model" + ".pth.tar" ))
        model.load_state_dict(args.checkpoint['model_state_dict'])
        args.current_epoch = args.checkpoint['epoch']
    else:
        args.current_epoch = 0
        logging.info('No checkpoint used')
    
    return model

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
    #try:
        #args = utils.load_args(args.experiment_dir, ARGS_FILE)
        #save each time
    #    pass
    #except:
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
    logging.info("Train loader ready")

    val_loader = get_loader(data_root=args.root_dir,
                              vocab=vocab,
                              batch_size=args.batch_size,
                              data_type='val',
                              shuffle=False,
                              num_workers=0,
                              debug=False)
    logging.info("Val loader ready")

    test_loader = get_loader(data_root=args.root_dir,
                              vocab=vocab,
                              batch_size=args.batch_size,
                              data_type='test',
                              shuffle=False,
                              num_workers=0,
                              debug=False)
    logging.info("Test loader ready")

    model = create_model(args, vocab, feature_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    print("args.opt: " + args.opt)
    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.899999976158, weight_decay=0.000500000023749)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if args.resume_epoch > 0:
        optimizer.load_state_dict(args.checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(args.checkpoint['scheduler_state_dict'])

    train_loss_array = []
    val_loss_array = []

    #some big number
    min_val_loss = 10000000

    train_epoch_array = []
    val_epoch_array = []

    logging.info(model)
    train(args, model, train_loader, val_loader, len(vocab))

    t0 = time.time()
    test_loss = test(args,model,test_loader, len(vocab), beam=None)
    logging.info("Testing done in: {:3.1f} seconds".format(time.time() - t0))
    
    #save_final_captions(args, model, test_loader, max_sent_len=12, beam_width=5)

if __name__ == "__main__":
    main()
