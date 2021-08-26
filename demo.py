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
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long() 
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(pcb_model,img):    
    image_feature = torch.FloatTensor(1,2048,6).zero_() 
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

    file_path = 'Entrance-1.mp4'
    video_capture = cv2.VideoCapture(file_path)

    writeVideo_flag = True

    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
 
    count = 0

    gallery_feature = np.array([])
    gallery_id = []
    current_id = 0
    person_id = -1
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        count = count + 1
        if(count % 10 != 1):
            continue

        t1 = time.time()

        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)
        image = np.asarray(image)

        for i in range(len(boxes)):
            x_start = boxes[i][1])
            x_end = boxes[i][1] + boxes[i][3]
            y_start = boxes[i][0]
            y_end = boxes[i][0] + boxes[i][2]

            detected_img = image[max(0 , x_start) : max(0 , x_end) , max(0 , y_start) : max(0 , y_end)]
            detected_img = Image.fromarray(detected_img)
             
            transformed_img = torch.reshape(data_transforms(detected_img) , (1 , 3 , 384 , 192))
            image_feature = extract_feature(model , transformed_img)
            image_feature = image_feature.numpy()
    
            num_values = image_feature.shape[1]
            
            if gallery_feature.size == 0:
               gallery_feature = np.concatenate((gallery_feature , image_feature) , axis = 0)
               current_id = current_id + 1
               gallery_id.append(current_id)
               person_id = current_id
            else:
               rms_error = rms_error = np.sqrt(np.sum((gallery_feature - image_feature)**2 , axis = 1)/num_values)
               if(min(rms_error) <= threshold):
                   id_index = np.argmin(rms_error)
                   person_id = gallery_id[id_index]               
               else:
                   gallery_feature = np.concatenate((gallery_feature , image_feature) , axis = 0) 
                   current_id = current_id + 1
                   gallery_id.append(current_id)
                   person_id = current_id

             cv2.rectangle(frame, (y_start, x_start), (y_end, x_end), (255, 255, 255), 2)
             cv2.putText(frame, "ID: " + str(person_id), (y_start, x_start), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 0), 1)

        cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))
        print('\n')
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if(count >= 301):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

 
    video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())







