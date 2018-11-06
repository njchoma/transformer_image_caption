"""
Author: Omkar Damle
Date: 28th Oct 2018

Dataloader class
Points to note:
1. The annotations are returned as an array of indices according to the vocabulary dictionary
2. Each annotation starts with the start symbol and ends with the end symbol. There can be padding after the end symbol in order
to make a batch
3. Each image feature has the shape - (36,2048)

reference: 
https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""


import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from vocab import Vocabulary
from pycocotools.coco import COCO

import h5py
import argparse
import cPickle

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, data_root, vocab, data_type, debug=False):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            data_root: root dir.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.data_type = data_type
        self.imgid2idx = cPickle.load(open(os.path.join(data_root, data_type + "36_imgid2idx.pkl"), 'rb'))
        if debug:
            file_name = data_root + '/' + data_type + "36_mini.hdf5"
            with open(os.path.join(data_root,"mini_anno.pkl"), 'rb') as f:
                self.coco = pickle.load(f)
        else:
            file_name = data_root + '/' + data_type + "36.hdf5"      
            annotation_path = os.path.join(data_root,
                                           "captions",
                                           "annotations",
                                           "captions_"+data_type+"2014.json")
            self.coco = COCO(annotation_path)


        data_h5 = h5py.File(file_name,'r')       
        self.train_features = np.array(data_h5.get('image_features'))
        
        ids = list(self.coco.anns.keys())

        self.data_root = data_root
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        #print('Inside getitem, Retrieving for index: ' + str(index))
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        #path = coco.loadImgs(img_id)[0]['file_name']
        index = self.imgid2idx[img_id]
        features = self.train_features[index]
        features = torch.Tensor(features)
        # Convert caption (string) to word ids.
        #print('converting captions to word ids')
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        vocab_len = len(vocab)
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        #print(len(caption))
        target = torch.Tensor(caption)
        return features, target

    def __len__(self):
        return len(self.ids)

"""
def indexto1hot(vocab_len, index):
    one_hot = np.zeros([vocab_len])
    one_hot[index] = 1
    return one_hot
"""

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - feature: torch tensor of shape (36,2048).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        features: torch tensor of shape (batch_size, 36, 2048).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    print("Length of list: " + str(len(data)))

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    features, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    features = torch.stack(features, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return features, targets, lengths

def get_loader(data_root, vocab, batch_size, data_type, shuffle, num_workers, debug=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(data_root=data_root,
                       vocab=vocab, data_type = data_type, debug=debug)
    
    # Data loader for COCO dataset
    # This will return (features, captions, lengths) for each iteration.
    # features: a tensor of shape (batch_size, 36, 2048).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn = collate_fn)
    return data_loader

