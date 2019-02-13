#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 14:45:16 2019

@author: akramkohansal
"""

import os
import cv2
import string
from tqdm import tqdm
import click
import numpy as np
from models.crnn import CRNN
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
from dataset.load_data import LoadDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from dataset.data_transform import Resize, Rotation, Translation, Scale
from models.model_loader import load_model
from torchvision.transforms import Compose
from collections import OrderedDict

from PIL import Image, ImageDraw, ImageFont


def detect(net, data, cuda, visualize, batch_size=1):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)
    
    print(data_loader)
    
    print(len(data_loader))
    iterator = tqdm(data_loader)
    print(len(iterator))
    for sample in iterator:
      
        imgs = Variable(sample["img"])
       
        print(sample["imgname"])
        if cuda:
            imgs = imgs.cuda()
        out = net(imgs, decode=True)
        gt = (sample["seq"].numpy() - 1).tolist()
        lens = sample["seq_len"].numpy().tolist()
        pos = 0
        for i in range(len(out)):
            print("in detection")
            print(out[i])
            gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])
            print("gts is ----")
            print(gts)
            print("detected is")
            print(out[i])
            img = imgs[i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
            cv2.imshow("img", img)
            key = chr(cv2.waitKey() & 255)
            if key == 'q':
                break
    #return acc, avg_ed

def detect2(net, imgfolder, cuda, transform):
    
    imgefolder = "/home/akram/Documents/crnn_result/img3/"
    image_list = os.listdir(imgefolder)
    #var_name = None
    #img_count = 0
    
    for imgfile in image_list:
        #print(img_count)
        #img_count = img_count + 1
        #print(img_count)
        img_full_path = imgefolder+imgfile
        print(imgfile)
        #img1 = cv2.imread(img_full_path)
        img1 = Image.open(img_full_path).convert('RGB')
        #print(img_full_path)
        img = img1.resize((320, 32))
        if isinstance(img, Image.Image):
            #print("Image file")
            width = img.width
            height = img.height
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            
            #if var_name is None:
             #   print("is none")
              #  var_name = img
               # print(var_name)
                
                
            #else:
             #   print(torch.equal(img, var_name))
            
            img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
            img = img.view(1, 3, height, width)
            img = img.float().div(255.0)
        elif type(img) == np.ndarray: # cv2 image
            print("2")
            img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
        else:
            print("unknow image type")
            exit(-1)
       
        #imgs = Variable(img)
       
        if cuda:
            img = img.cuda()
            net = net.cuda()
        img = torch.autograd.Variable(img)
        #print(img)
        out = net(img, decode=True)
     
        for i in range(len(out)):
            print("in detection")
            print(len(out))
            print(out[i])
    
def load_weights(target, source_state):
    new_dict = OrderedDict()
    #for k, v in source_state.items():
       # print(k)
    print(len(source_state))
    print(len(target.state_dict()))
    for k, v in target.state_dict().items():
        
        #if k == "proj.weight":
         #   print("proj weight")
          #  print(v)
        #if k == "proj.bias":
         #   print("proj bias")
          #  print(v)    

        nk = "module."+k
        #if k == "proj.bias":
         #   print("proj bias")
          #  print(v)   
           # print(v.size())
            #print(source_state[nk].size())
            #print(source_state[nk])
        if nk in source_state and v.size() == source_state[nk].size():
            #print("injaaaa")
            new_dict[k] = source_state[nk]
        else:
            print(k)
            print("naaaaaaaaa injaaaa")
            print(source_state[nk])
            print(v)
            new_dict[k] = v
    target.load_state_dict(new_dict)

    

@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default=string.digits+string.ascii_letters, help='Alphabet')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')
def main(data_path, abc, seq_proj, backend, snapshot, input_size, gpu, visualize):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    cuda = True if gpu is not '' else False
    abc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    #print(abc)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Rotation(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        
        data = LoadDataset(data_path=data_path, mode="test", transform=transform)
        
    
    seq_proj = [int(x) for x in seq_proj.split('x')]
    
    #net = load_model(abc, seq_proj, backend, snapshot, cuda)
    net = CRNN(abc=abc, seq_proj=seq_proj, backend=backend)
    #net = nn.DataParallel(net)
    if snapshot is not None:
        load_weights(net, torch.load(snapshot))
    if cuda:
        net = net.cuda()
    #import pdb;pdb.set_trace()    
    net = net.eval()
    detect(net, data, cuda, visualize)
    #detect2(net, imgfolder=data_path,cuda=cuda, transform=transform)

if __name__ == '__main__':
    main()
