#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import os
import warnings
import cv2
import numpy as np
from PIL import Image
import scipy.misc
import pandas as pd

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import scipy.io
from model import ft_net, ft_net_dense, PCB, PCB_test

warnings.filterwarnings('ignore')

def load_network(network):
    save_path = os.path.join('./model','A','net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(pcb_model,img):    
    image_feature = torch.FloatTensor(1,2048,6).zero_() # we have four parts
    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        outputs = pcb_model(input_img) 
        f = outputs.data.cpu()
        image_feature = image_feature + f

    fnorm = torch.norm(image_feature, p=2, dim=1, keepdim=True)
    image_feature = image_feature.div(fnorm.expand_as(image_feature))
    image_feature = image_feature.view(image_feature.size(0), -1)

    return image_feature



class_num=751
torch.cuda.set_device(0)
use_gpu = torch.cuda.is_available()

model_structure = PCB(class_num)
model = load_network(model_structure)
model = PCB_test(model)
model = model.eval()
if use_gpu:
    model = model.cuda()

data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


df = pd.read_csv('detected_boxes_details.csv')
image_names = df['image_names']

gallery_feature = []
gallery_id = []
image_id = []
current_id = 0
person_id = -1

count = 0

for name in image_names:
    count = count + 1
    img = Image.open(name)
    img = torch.reshape(data_transforms(img) , (1 , 3 , 384 , 192))
    image_feature = extract_feature(model , img)
    image_feature = image_feature.numpy()

    num_values = image_feature.shape[1]
            
    if count == 1:
       gallery_feature.append(image_feature)
       gallery_feature = np.reshape(np.array(gallery_feature) , (1 , num_values))
       current_id = current_id + 1
       gallery_id.append(current_id)
       person_id = current_id
    else:
       rms_error = np.sqrt(np.sum((gallery_feature - image_feature)**2 , axis = 1)/num_values)
       if(min(rms_error) <= 0.02):
           id_index = np.argmin(rms_error)
           person_id = gallery_id[id_index]               
       else:
           gallery_feature = np.concatenate((gallery_feature , image_feature) , axis = 0) 
           current_id = current_id + 1
           gallery_id.append(current_id)
           person_id = current_id

    image_id.append(person_id)

df['person_id'] = image_id
df.to_csv('detected_boxes_details_with_id.csv' , index = False)















'''
image_name1 = 'detected-images/21_5.jpg'
image_name2 = 'detected-images/1_5.jpg'

img1 = Image.open(image_name1)
img2 = Image.open(image_name2)

img1 = torch.reshape(data_transforms(img1) , (1 , 3 , 384 , 192))
img2 = torch.reshape(data_transforms(img2) , (1 , 3 , 384 , 192))

image_feature1 = extract_feature(model , img1)
image_feature2 = extract_feature(model , img2)

image_feature1 = image_feature1.numpy()
image_feature2 = image_feature2.numpy()

num_values = image_feature1.shape[1]

rms_error = np.sqrt(np.sum((image_feature1 - image_feature2)**2 , axis = 1)/num_values)
print(rms_error)
'''
