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
from torch.nn.utils.rnn import pack_padded_sequence

import utils_experiment as utils
from data_helpers.data_loader_ks import get_loader
from data_helpers.vocab import Vocabulary

# from eval.coco_caption.eval import evaluate

from models.caption_model import Caption_Model
from models.simple_model import Simple_Model
from models.transformer import Transformer

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
        len_captions = len(captions[0])
        if torch.cuda.is_available():
            features, captions = features.cuda(), captions.cuda()

        out = model(features, len_captions, captions)
        n_ex, vocab_len = out.view(-1, len_vocab).shape

        captions = captions[:,1:]
        decode_lengths = [x-1 for x in lengths]
        captions,_ = pack_padded_sequence(captions,
                                          decode_lengths,
                                          batch_first=True)
        out,_ = pack_padded_sequence(out, decode_lengths,batch_first = True)

        batch_loss = loss(out,captions)
        epoch_loss+=batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()        

    epoch_loss = epoch_loss/nb_batch 
    logging.info("Train loss: {:>.3E}".format(epoch_loss))
    return epoch_loss

def val_one_epoch(args, model, val_loader, len_vocab, vocab, beam=None):
    model.eval()
    nb_batch = len(val_loader)
    nb_val = nb_batch * args.batch_size
    logging.info("Validating {} batches, {} samples.".format(nb_batch, nb_val))

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0

    gts = {}
    res = {}

    with torch.no_grad():
        for i, (image_ids, features, captions, lengths) in enumerate(val_loader):
            if (i % (nb_batch//20)) == 0:
                logging.info("  {:5d}".format(i))
            len_captions = len(captions[0])

            #serial part. Will not slow done much as size of val dataset is only 5k

            if torch.cuda.is_available():
                features, captions = features.cuda(), captions.cuda()

            if beam is not None:
                if (i % 200) == 0:
                    sentences = model(features, 20, beam)
                    print(sentences)

            out = model(features, len_captions, captions)
            
            '''
            for ii, image_id in enumerate(image_ids):    
                caption_list = [vocab.get_word(wid.item()) for wid in captions[ii] if wid!= 0]                
                caption_list = caption_list[1:-2]
                caption = ' '.join(caption_list)
                padded_result = [vocab.get_word(torch.argmax(one_hot_en).item()) for one_hot_en in out[ii]]
                result_list = []
                for word in padded_result:
                    if word == '.' or word == '<end>':
                        break
                    result_list.append(word)
                result = ' '.join(result_list)
                if image_id in gts:
                    gts[image_id].append({'image_id': int(image_id), 'caption': caption})
                else:
                    gts[image_id] = [{'image_id': int(image_id), 'caption': caption}]

                res[image_id] = [{'image_id': int(image_id), 'caption': result}]

            '''
            n_ex, vocab_len = out.view(-1, len_vocab).shape
            captions = captions[:,1:]
            
            decode_lengths = [x-1 for x in lengths]
            captions,_ = pack_padded_sequence(captions,
                                              decode_lengths,
                                              batch_first=True)
            out,_ = pack_padded_sequence(out, decode_lengths,batch_first = True)
            batch_loss = loss(out,captions)
            epoch_loss+=batch_loss.item()
    bleu4_score = 0 # evaluate(gts,res)
    logging.info("BLEU score computed: " + str(bleu4_score))
    epoch_loss = epoch_loss/nb_batch
    logging.info("Val loss: {:>.3E}".format(epoch_loss))
    return bleu4_score, epoch_loss

def train(args, model, train_loader, val_loader, optimizer, scheduler, len_vocab, vocab):
    logging.warning("Beginning training")

    train_loss_array = []
    val_loss_array = []
    val_bleu4_array = []
    #some big number
    min_val_loss = 10**5
    max_bleu_score = 100
    train_epoch_array = []
    val_epoch_array = []
    
    if args.current_epoch == 0:
        bleu4_score, val_loss = val_one_epoch(args,model,val_loader, len_vocab, vocab, beam=None)
        logging.info("Validation loss with random initialization. Loss: " + str(val_loss) + ", BLEU4 score: " + str(bleu4_score))
    
    logging.info("Maximum of epochs: " + str(args.max_nb_epochs))

    while args.current_epoch < args.max_nb_epochs:
        args.current_epoch += 1
        logging.info("\nEpoch {}".format(args.current_epoch))
        
        t0=time.time()
        train_loss = train_one_epoch(args, model, train_loader, optimizer, len_vocab)
        logging.info("Train done in: {:3.1f} seconds".format(time.time() - t0))
        
        t0 = time.time()
        bleu4_score, val_loss = val_one_epoch(args,model,val_loader, len_vocab, vocab, beam=None)
        logging.info("Valid done in: {:3.1f} seconds".format(time.time() - t0))

        scheduler.step(val_loss)

        torch.save({
                    'epoch': args.current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    },
                    os.path.join(args.experiment_dir,
                             "epoch_" + str(args.current_epoch) + ".pth.tar"))

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)
        val_bleu4_array.append(bleu4_score)
        train_epoch_array.append(args.current_epoch)
        val_epoch_array.append(args.current_epoch)

        #keep track of the best model and save it
        if bleu4_score > max_bleu_score:
            max_bleu_score = bleu4_score
            torch.save({
                'epoch': args.current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(args.experiment_dir, "best_model"  + ".pth.tar"))
    plt.figure(0)
    plt.plot(train_epoch_array, train_loss_array, label='Training loss')
    plt.plot(val_epoch_array, val_loss_array, label = 'Validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.experiment_dir, "loss_stats.png"))

    plt.figure(1)
    plt.plot(val_epoch_array, val_bleu4_array, label = 'Validation BLEU score')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('BLEU score')
    plt.savefig(os.path.join(args.experiment_dir, "bleu_stats.png"))
    

def create_model(args, vocab, feature_dim):
    model = None
    #teacher_forcing ratio
    tf_ratio = args.teacher_forcing
    logging.info("Teacher forcing ratio: {:.2f}".format(tf_ratio))
    
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
        model = Transformer(dict_size=len(vocab),
                              image_feature_dim=feature_dim,
                              vocab=vocab,
                              tf_ratio=tf_ratio)
        logging.info("Transformer model created.")
    else:
        logging.error("Model type {} not understood".format(args.model_type))
    
    if args.resume_epoch > 0:
        logging.info('Loading checkpoint')
        
        args.checkpoint = torch.load(os.path.join(args.experiment_dir,
                            "epoch_" + str(args.resume_epoch) + ".pth.tar" ))
        model.load_state_dict(args.checkpoint['model_state_dict'])
        args.current_epoch = args.resume_epoch
    elif args.resume_epoch < 0:
        logging.info('Loading best checkpoint')
        args.checkpoint = torch.load(os.path.join(args.experiment_dir,
                                                "best_model" + ".pth.tar" ))
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

    test_batch_size = args.batch_size
    if args.beam_search:
        test_batch_size = 1
    test_loader = get_loader(data_root=args.root_dir,
                              vocab=vocab,
                              batch_size= test_batch_size,
                              data_type='test',
                              shuffle=False,
                              num_workers=0,
                              debug=False)
    logging.info("Test loader ready")

    model = create_model(args, vocab, feature_dim)
    if torch.cuda.is_available():
        model = model.cuda()
        logging.info("GPU type:\n{}".format(torch.cuda.get_device_name(0)))

    logging.info("args.opt: " + args.opt)
    optimizer = None
    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum = 0.899999976158,
                              weight_decay=0.000500000023749)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if args.resume_epoch > 0:
        optimizer.load_state_dict(args.checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(args.checkpoint['scheduler_state_dict'])

    logging.info(model)
    train(args,
          model,
          train_loader,
          val_loader,
          optimizer,
          scheduler,
          len(vocab), 
          vocab)

if __name__ == "__main__":
    main()
