from data_loader import get_loader
from vocab import Vocabulary
import pickle
import time
import argparse
import os

def main(args):
    with open(os.path.join(args.root_dir, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    data_type = args.data_type
    data_loader = get_loader(args.root_dir, vocab, args.batch_size, data_type, shuffle=True, num_workers = args.num_workers)

    print('Iterating the dataset')
    print("Length of data loader: " + str(len(data_loader)))
    for i, (features, captions, lengths) in enumerate(data_loader):
        print("Index: " + str(i))
        #print("Features shape: ")
        #print(features.shape)
        print("Captions shape: ")
        print(captions.shape)
        print("Lengths: ")
        print(len(lengths))
        #time.sleep(10)
        print(captions[0])
        print(captions[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='', help='root directory')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_type', type=str, default='train')
    args = parser.parse_args()
    print(args)
    main(args)
