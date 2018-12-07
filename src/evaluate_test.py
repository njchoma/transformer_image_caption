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

def save_final_captions(args, model, val_loader, vocab, max_sent_len, beam_width=None):
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
            prev_word_id = -1
            prev_word_ids = []
            if beam_width == None:
                sentence = sentence.squeeze()
                #sentence = [ vocab.get_word(torch.nonzero(onehotarray)[0][0]) for onehotarray in sentence]
                sentence1 = []
                correct_sentence = []
                #print(captions)
                #print(captions[0])
                for i in range((len(captions[0]))):
                    correct_sentence.append(vocab.get_word(captions[0][i].item()))
                for i in range(len(sentence)):
                    curr_word_id = torch.argmax(sentence[i]).item()
                    
                    """
                    if curr_word_id == prev_word_id:
                        _, indices = torch.topk(sentence[i],2)
                        second_max_index = indices[1]
                        curr_word_id = second_max_index.item()
                    """
                    if curr_word_id in prev_word_ids:
                        _,indices = torch.topk(sentence[i],20)
                        top_count = 0
                        while curr_word_id in prev_word_ids:
                            top_count += 1
                            curr_word_id = indices[top_count].item()
                            
                    
                    sentence1.append(vocab.get_word(curr_word_id))
                    #prev_word_id = curr_word_id
                    prev_word_ids.append(curr_word_id)
                print("Correct sentence: ")
                print(correct_sentence)
                print("Predicted: ")
                print(sentence1)
                s.add_sentence(image_ids[0], sentence1)
            else:
                s.add_sentence(image_ids[0], sentence[1])
            if (i % 50) == 0:
                logging.info("  {:4d}".format(i))
    logging.info("Saving sentences...")
    s.save_sentences()
    logging.info("Done.")


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
    
    save_final_captions(args, model, test_loader, vocab, max_sent_len=12)
    #save_final_captions(args, model, test_loader, vocab, max_sent_len=12, beam_width=5)
    test_loss = test(args,model,test_loader, len(vocab), beam=None)
    logging.info("Testing done in: {:3.1f} seconds".format(time.time() - t0))

if __name__ == "__main__":
    main()
