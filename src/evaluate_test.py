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
from torch.nn.utils.rnn import pack_padded_sequence

import utils_experiment as utils
from data_helpers.data_loader_ks import get_loader
from data_helpers.vocab import Vocabulary
from models.caption_model import Caption_Model
from models.simple_model import Simple_Model

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

def test(args, model, val_loader, len_vocab, beam=None):
    model.eval()
    nb_batch = len(val_loader)
    nb_val = nb_batch * args.batch_size
    logging.info("Testing {} batches, {} samples.".format(nb_batch, nb_val))

    loss = torch.nn.CrossEntropyLoss()
    epoch_loss = 0

    with torch.no_grad():
        for i, (image_ids,features,captions,lengths) in enumerate(val_loader):
            len_captions = len(captions[0])
            if torch.cuda.is_available():
                features, captions = features.cuda(), captions.cuda()

            if beam is not None:
                if (i % 200) == 0:
                    sentences = model(features, 20, beam)
                    print(sentences)
            out = model(features, len_captions, captions)
            n_ex, vocab_len = out.view(-1, len_vocab).shape
            captions = captions[:,1:]
            
            decode_lengths = [x-1 for x in lengths]
            captions,_ = pack_padded_sequence(captions,
                                              decode_lengths,
                                              batch_first=True)
            out,_ = pack_padded_sequence(out, decode_lengths, batch_first=True)

            batch_loss = loss(out,captions)
            epoch_loss+=batch_loss.item()
   
    epoch_loss = epoch_loss/nb_batch
    logging.info("Test loss: {:>.3E}".format(epoch_loss))
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
            sentence = model(features, max_sent_len, None, beam_width)
            s.add_sentence(image_ids[0], sentence[1])
            if (i % 500) == 0:
                logging.info("  {:4d}".format(i))
    logging.info("Saving sentences...")
    s.save_sentences()
    logging.info("Done.")

def create_model(args, vocab, feature_dim):
    model = None
    
    tf_ratio = 1.0
    if args.model_type == 'bottom_up':
        model = Caption_Model(dict_size=len(vocab),
                              image_feature_dim=feature_dim,
                              vocab=vocab,
                              tf_ratio=tf_ratio)
        logging.info("Bottom-Up model created.")
    elif args.model_type == 'simple':
        model = Simple_Model(dict_size=len(vocab),
                              image_feature_dim=feature_dim,
                              vocab=vocab,
                              tf_ratio=tf_ratio)
        logging.info("Simple model created.")
    elif args.model_type == 'transformer':
        logging.error("Transformer model not yet implemented")
        exit()
    
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

    args.current_epoch = 0
    utils.save_args(args, args.experiment_dir, ARGS_FILE)


    logging.info("Loading data...")
    with open(os.path.join(args.root_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

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

    logging.info(model)

    t0 = time.time()
    
    save_final_captions(args, model, test_loader, max_sent_len=12, beam_width=args.beam_width)
    test_loss = test(args, model, test_loader, len(vocab), beam=None)
    logging.info("Testing done in: {:3.1f} seconds".format(time.time() - t0))

if __name__ == "__main__":
    main()
