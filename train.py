from __future__ import print_function
import argparse
from math import log10
from random import randrange
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from epcarn import Net as RCARN
from data import get_training_set
import pdb
import socket
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import cv2
from canny.canny import * 

model_type = 'RCARN'
nFrames = 4
upscale_factor = 2
gpus = 2
lr = 1e-4
nEpochs = 150
start_epoch = 1
cuda = True
gpus_list = range(gpus)
residual = False
prefix = 'EPCARN'
save_folder = './checkpoints/'
hostname = str(socket.gethostname())
save_point = './checkpoints/2x_XPRIZE_AI_4RCARN_Recurrent_interpolation_epoch_28.pth'

batch_size = 1

output = './Results/'

data_dir = './dataset/video'

#loss_hist = [0]*1000
plt_epoch = []
plt_loss = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ ==  '__main__':
    def save_model(filename):
        torch.save(model.state_dict(), './checkpoints/' + filename)

    print('===> Loading datasets')
    train_set = get_training_set(data_dir = data_dir, 
                                nFrames = nFrames, 
                                upscale_factor = 2, 
                                data_augmentation = True, 
                                other_dataset = False, 
                                patch_size = 128, 
                                future_frame = False)
    training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1, shuffle=False)

    print('===> Building model ', model_type)
    if model_type == 'RCARN':
        model = RCARN(num_channels=3, 
                    base_filter=256,  
                    feat = 3,
                    num_stages=3, 
                    n_resblock=5, 
                    nFrames=nFrames, 
                    scale_factor=upscale_factor).to(device)

    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    def load_checkpoint():
        # Continue from last checkpoint
        start_epoch = 9
        model_c = save_point
        model.load_state_dict(torch.load(model_c, map_location=lambda storage, loc: storage))
        print('Checkpoint SR(RCARN) model is loaded.')

    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)
        
    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    def save_loss(loss, epoch, step):
        #loss_hist[-1] += loss
        plt_epoch.append(epoch)
        plt_loss.append(loss)
        if epoch % step == 0:
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(plt_epoch, plt_loss)
            plt.savefig('EPCARN' + prefix +'_Loss.png')
             

    def checkpoint(epoch):
        model_out_path = save_folder+str(upscale_factor)+'x_'+hostname+model_type+prefix+"_epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def load(self, path):

        states = torch.load(path)
        state_dict = self.refiner.state_dict()
        for k, v in states.items():
            if k in state_dict.keys():
                state_dict.update({k: v})
        self.refiner.load_state_dict(state_dict)


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
    

    def save_img_pre(img, img_name, pred_flag):
        save_img = img.data[0].numpy().transpose(1,2,0)

        output = './Results/Predict'
        save_dir = output
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if pred_flag:
            save_fn = save_dir +'/'+'N_'+model_type+'F'+str(nFrames)+'N'+ img_name +'.png'
        else:
            save_fn = save_dir +'/'+ img_name+'.png'
        cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def save_img_bic(img, img_name, pred_flag):
        save_img = img.data[0].numpy().transpose(1,2,0)
        #print(save_img.size())

        # save img
        output = './Results/Bicubic'
        save_dir = output
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if pred_flag:
            save_fn = save_dir +'/'+'N_'+model_type+'F'+str(nFrames)+'N'+ img_name +'.png'
        else:
            save_fn = save_dir +'/'+ img_name+'.png'
        cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])


    def save_img_tar(img, img_name, pred_flag):
        save_img = img.data[0].numpy().transpose(1,2,0)
        #print(save_img.size())

        # save img
        output = './Results/Target'
        save_dir = output
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if pred_flag:
            save_fn = save_dir +'/'+'N_'+model_type+'F'+str(nFrames)+'N'+ img_name +'.png'
        else:
            save_fn = save_dir +'/'+ img_name+'.png'
        cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

    load_checkpoint()
    start_epoch = 29
    for epoch in range(start_epoch, 100 + 1):
        epoch_loss = 0
        total_time = 0

        print('Epoch ', epoch, 'Training!')
        
        for iteration, batch in enumerate(training_data_loader, 1):
            input, target, bicubic = batch[0], batch[1], batch[2]

            if cuda:
                input = Variable(input).to(device)
                target = Variable(target).to(device)
                bicubic = Variable(bicubic).to(device)
                canned = edge_detect(target)

            optimizer.zero_grad()
            t0 = time.time()
            prediction = model(input, canned)

            loss = criterion(prediction, target)
            t1 = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_time += (t1 - t0)
            
            if iteration % 1 == 1000:
                
                save_img_pre(prediction.cpu().data, str(iteration), False)
                save_img_bic(bicubic.cpu().data, str(iteration), False)
                save_img_tar(target.cpu().data, str(iteration), False)
        
        avg_loss = epoch_loss / len(training_data_loader)
        try:
            save_loss(loss=avg_loss, epoch=epoch, step=1)
        except:
            print('Error Save')
        checkpoint(epoch)
        print("===> Epoch {} Complete: Avg. Loss: {:.4f} || Avg. Time: {:.4f}".format(epoch, epoch_loss / len(training_data_loader), total_time / len(training_data_loader)))

    