from collections import OrderedDict

import torch
from torch import nn

from .crnn import CRNN

def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        #print(k)
        
        #print("in loading model")
        nk = "module."+k
        if nk in source_state and v.size() == source_state[nk].size():
            new_dict[k] = source_state[nk]
        else:
            new_dict[k] = v
    target.load_state_dict(new_dict)

def load_model(abc, seq_proj=[0, 0], backend='resnet18', snapshot=None, cuda=True):
    net = CRNN(abc=abc, seq_proj=seq_proj, backend=backend)
    #net = nn.DataParallel(net)
    if snapshot is not None:
        load_weights(net, torch.load(snapshot))
    if cuda:
        net = net.cuda()
    return net
