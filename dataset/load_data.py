#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:21:12 2019

@author: akramkohansal
"""

from torch.utils.data import Dataset
import json
import os
import cv2

class LoadDataset(Dataset):
    def __init__(self, data_path, mode="recog", transform=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        #self.config = json.load(open(os.path.join(data_path, "desc.json")))
        self.transform = transform

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "test":
            return int(len(self.config[self.mode]) * 0.01)
        return len(self.config[self.mode])

    def __getitem__(self, idx):
        name = self.config[self.mode][idx]["name"]

        img = cv2.imread(os.path.join(self.data_path, name))
        sample = {"img": img, "aug": self.mode == "recog"}
        if self.transform:
            sample = self.transform(sample)
        return sample
