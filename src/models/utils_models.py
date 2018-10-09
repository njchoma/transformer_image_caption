import torch
import torch.nn as nn

def make_zeros(shape, cuda=False):
    zeros = torch.zeros(shape)
    if cuda:
        zeros = zeros.cuda()
    return zeros
