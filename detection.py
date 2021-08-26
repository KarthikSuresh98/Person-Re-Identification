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
from yolo import YOLO
import pandas as pd

from tools import generate_detections as gdet
import imutils.video

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
    image_feature = torch.FloatTensor(n,2048,6).zero_() # we have four parts
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

def main(yolo):

    file_path = 'Entrance-1.mp4'      # Path of the data file
    video_capture = cv2.VideoCapture(file_path)

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
 
    count = 0
 
    image_names = []
    bounding_boxes = []
    frame_names = []

    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        count = count + 1
        if(count % 10 != 1):
            continue

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        frame_name = 'video-frames/frames/' + str(count) + '.jpg'
        image.save(frame_name)
        boxes, confidence, classes = yolo.detect_image(image)
        image = np.asarray(image)

        for i in range(len(boxes)):
            x_start = boxes[i][1]
            x_end = boxes[i][1] + boxes[i][3]
            y_start = boxes[i][0]
            y_end = boxes[i][0] + boxes[i][2]

            detected_image = image[max(0 , x_start) : max(0 , x_end) , max(0 , y_start) : max(0 , y_end)]

            bounding_box = [x_start , x_end , y_start , y_end]

            frame_names.append(frame_name)

            detected_image = Image.fromarray(detected_image)
            image_name = 'detected-images/' + str(count) + '_' + str(i+1) + '.jpg'
            image_names.append(image_name)
            bounding_boxes.append(bounding_box)
            detected_image.save(image_name)
            


        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))
        print('\n')
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if(count >= 100):
            break
    data = {'image_names' : image_names , 'bounding_boxes' : bounding_boxes , 'frame_names' : frame_names}
    df = pd.DataFrame(data)
    df.to_csv('detected_boxes_details.csv' , index = False)

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))


    video_capture.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
  
    torch.cuda.set_device(0)
    main(YOLO())


 
