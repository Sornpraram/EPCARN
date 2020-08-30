from __future__ import print_function
import sys
import numpy as np
import pyzed.sl as sl
import cv2

import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from epcarn import Net as RCARN
from data import get_test_set
import pdb
import socket
import time
from tqdm.notebook import tqdm

import numpy as np
import math
from functools import reduce
import pdb
import scipy.io as sio

from canny.canny import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "./"

def main() :

    # Prepare ZED camera only
    # For orther camera can ignore this section
    zed = sl.Camera()
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    key = ' '
    # End ZED setup

    def edge_detect(hr):
        canned = []
        if 1:
            for hr_i in hr:
                canned.append(canny(hr_i.unsqueeze(0).half(),use_cuda=True))
            canned = torch.cat(canned,dim=0).detach().float()
            EDGE_SIZE = 1
            canned[:,:,:EDGE_SIZE,:] = 0
            canned[:,:,-EDGE_SIZE:,:] = 0
            canned[:,:,:,:EDGE_SIZE] = 0
            canned[:,:,:,-EDGE_SIZE:] = 0
        return canned

    def customTransform(input,channel = 3):
        im = np.array([input])
        if channel != 3:
            im = im[:,:,:,:channel]
        im = np.transpose(im, (0, 3, 1, 2))
        im = np.array(im, dtype=np.float32)  / 255.0
        im = torch.FloatTensor(im)
        im = im.cuda()
        return im

    def show_img(ims,wait=0,names=['im']):
        for im,name in zip(ims,names):
            try:
                temp = im[0].cpu().numpy()
            except:
                temp = im[0].cpu().detach().numpy()
            temp = np.clip(np.transpose(temp, axes=(1, 2, 0)),0,1.0)*255
            temp = temp.astype(np.uint8)
            cv2.imshow(name, temp)
        cv2.waitKey(wait)

    def rescale_img(img_in, scale):
        img_in = img_in[:,:,:3]
        frame_HR = cv2.resize(img_in,(640,360))
        frame_LR = cv2.resize(frame_HR,(320,180))
        Show_LR = cv2.resize(frame_LR,(640,360))
        
        return frame_HR, frame_LR, Show_LR

    model = RCARN(num_channels=3, 
                    base_filter=256,  
                    feat = 3, #64
                    num_stages=3, 
                    n_resblock=5, 
                    nFrames=4, 
                    scale_factor=4).to(device)
    model_c = './checkpoints/EPCARN.pth'
    model.load_state_dict(torch.load(model_c, map_location=lambda storage, loc: storage))

    # Update...

    # .........

    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            image_ocv = image_zed.get_data()
            HR, LR, Show_LR = rescale_img(image_ocv, 2)

            #-------------------------------------------------------------
            img_send_jpeg = cv2.imencode(".jpg", LR, [cv2.IMWRITE_JPEG_QUALITY, 100])[1]
            nparr = np.asarray(img_send_jpeg, np.uint8)
            LR = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            #-------------------------------------------------------------
            HR = customTransform(HR)
            LR = customTransform(LR)
            Show_LR = customTransform(Show_LR)
            canned = edge_detect(HR)

            prediction = model(LR, canned)

            show_img([prediction,HR,Show_LR],wait=1,names=['sr','hr','lr'])


            key = cv2.waitKey(10)

    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()