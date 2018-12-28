import os
import json
import yaml
import logging
import argparse

#####################################
#               UTILS               #
#####################################
def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    # EXPERIMENT 
    add_arg('--root_dir', type=str, required=True,
            help='Folder where train, val, test data is stored')
    add_arg('--artifacts_dir', type=str, required=True,
            help='Folder where all training artifacts are saved')
    add_arg('--name', type=str, required=True,
            help='Name of current training scheme')
    add_arg('--run_nb', type=int, default=0,
            help='Run number of current iteration of model')
    add_arg('--debug', action='store_true',
            help='Set flag to load small dataset')
    add_arg('--beam_search', action='store_true',
            help='Set flag to use beam search while evaluating')
    add_arg('--model_type', type=str, default="bottom_up",
            help='Model type')
    add_arg('--beam_width', type=int, default=5,
            help='Beam width (used in evaluation)')
    # TRAINING
    add_arg('--batch_size', type=int, default=1,
            help='Minibatch size')
    add_arg('--resume_epoch', type=int, default=0,
            help='resume epoch number (0 if starting from scratch)')
    add_arg('--max_nb_epochs', type=int, default=1000,
            help='Maximum number of epochs to train') 
    add_arg('--lr', type=float, default=0.001,
            help='Learning rate')
    add_arg('--opt', type=str, default="Adam",
            help='Optimization method')
    add_arg('--teacher_forcing', type=float, default=0,
            help='teacher forcing ratio')
    

    return parser.parse_args()

def initialize_logger(experiment_dir):
    logfile = os.path.join(experiment_dir, 'log.txt')
    add_arg()
    # TRAINING
    add_arg('--batch_size', type=int, default=1,
            help='Minibatch size')
    add_arg('--resume_epoch', type=int, default=0,
            help='resume epoch number (0 if starting from scratch)')
    add_arg('--max_nb_epochs', type=int, default=1000,
            help='Maximum number of epochs to train') 
    add_arg('--lr', type=float, default=0.001,
            help='Learning rate')
    add_arg('--opt', type=str, default="Adam",
            help='Optimization method')
    add_arg('--teacher_forcing', type=float, default=0,
            help='teacher forcing ratio')
    

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

#########################################################
#                   SENTENCE OUTPUTS                    #
#########################################################
class Sentences(object):
    def __init__(self, savedir):
        self.sentences = []
        if os.path.exists(savedir) == False:
            os.makedirs(savedir)
        self.filepath = os.path.join(savedir, "final_sentences.json")

    def add_sentence(self, image_id, sentence):
        caption = ' '.join(sentence[1:-1])
        s = {'image_id':image_id, 'caption':caption}
        if (image_id % 200) == 0:
            print(s)
        self.sentences.append(s)

    def save_sentences(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.sentences, f)
